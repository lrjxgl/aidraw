import gradio as gr
import numpy as np
from PIL import Image
import os
import time
#文生图
from diffusers import StableDiffusionPipeline
import torch
import torch.cuda as cuda
from lib.fanyi_opus import zh2en
import argparse
parser = argparse.ArgumentParser()

parser.add_argument(
    "-e",
    "--engine",
    type=str,
    required=False,
    default="majicMIX_realistic_v6",
    help="翻译引擎",
)
args = parser.parse_args() 


if args.engine=="sdxl":
    model_id="stabilityai/stable-diffusion-xl-base-1.0"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        use_safetensors=True, variant="fp16"
        
    )
    pipe = pipe.to("cuda")
elif args.engine=="lcmsdxl":
    from diffusers import LCMScheduler, AutoPipelineForText2Image
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    adapter_id = "latent-consistency/lcm-lora-sdxl"
    pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")

    # load and fuse lcm lora
    pipe.load_lora_weights(adapter_id)
    pipe.fuse_lora()
elif args.engine=="majicMIX_realistic_v6":
    model_id = "digiplay/majicMIX_realistic_v6"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16      
    )
    pipe = pipe.to("cuda")
else:
    model_id = args.engine
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16      
    )
    pipe = pipe.to("cuda")
 

with gr.Blocks() as demo:
    gr.Markdown("""
    # 文生图
    """)
     
    with gr.Tab("文图生成"):
        with gr.Row():
            fytext=gr.Textbox(label="要翻译的内容",scale=3,placeholder="请输入要翻译的内容")    
            fyBtn=gr.Button("翻译",size="small")
        text=gr.Textbox(label="提示词",placeholder="请输入提示词")
    
        nprompt=gr.Textbox(label="反向提示词",value="ng_deepnegative_v1_75t,(badhandv4:1.2),(worst quality:2),(low quality:2),(normal quality:2),lowres,bad anatomy,(bad hands),watermark,many fingers,(broken hands),nsfw,EasyNegative,skin blemishes,(ugly:1.331),(duplicate:1.331),(morbid:1.21),(mutilated:1.21),mutated hands,(poorly drawn hands:1.5),(bad anatomy:1.21),(bad proportions:1.331),extra limbs,(disfigured:1.331),(extra legs:1.331),(fused fingers:1.61051),(too many fingers:1.61051),(unclear eyes:1.331),lowers,bad hands,missing fingers,extra digit,bad hands,missing fingers,(((extra arms and legs)))")
        with gr.Row():
            picnum=gr.Slider(minimum=1,maximum=9,step=1,value=4,label="图片数量")
            num_inference_steps=gr.Slider(minimum=1,maximum=50,step=1,value=30,label="步数")

        with gr.Row():
            width=gr.Slider(minimum=512,maximum=1024,step=64,value=512,label="宽度")
            height=gr.Slider(minimum=512,maximum=1024,step=64,value=512,label="高度")
        galler=gr.Gallery(label="生成的图片",columns=4)
        with gr.Row():
            submit=gr.Button("提交")
            saveBtn=gr.Button("保存图片",size="small")
        imgList=[] 
        def text2img(
                prompt,
                negative_prompt,
                picnum,
                width,
                height,
                num_inference_steps
        ):
            imgList.clear()
            for i in range(picnum):
                image = pipe(
                            prompt,
                            num_inference_steps=num_inference_steps,
                            width=width,
                            height=height,
                            guidance_scale=7.5,
                            negative_prompt=negative_prompt
                        ).images[0]
                cuda.empty_cache()
                imgList.append(image)
                yield imgList
            #return image
                
        def fanyi(text):            
            return zh2en(text)
        def saveImgList():
            t=time.time()
            print(imgList)
            for i in range(len(imgList)): 
                      
                imgList[i].save("./output/"+str(t)+"-"+str(i)+".png")
            gr.Info('保存成功')
        submit.click(text2img,inputs=[text,nprompt,picnum,width,height,num_inference_steps],outputs=galler)
        fyBtn.click(fanyi,inputs=fytext,outputs=fytext)
        saveBtn.click(saveImgList,inputs=None,outputs=None)
    
    with gr.Tab("图片列表") as tab2:
        

        def getImgs():
            dir="output"
            # 读取目录下所有文件 如果是图片(jpg,png) 则保存到imgs 列表中
            imgs = []
            for file in os.listdir(dir):
                if file.endswith("jpg") or file.endswith("png"):
                    imgs.append(os.path.join(dir, file))
            return imgs
        imgList2=getImgs()
            
        imgs = gr.State(value=imgList2)
        gallery = gr.Gallery(value=imgList2,allow_preview=False,columns=6)
        with gr.Row():
            selected = gr.Number(show_label=False,visible=False)
            show_btn = gr.Button("查看图片")
            new_btn = gr.Button("刷新列表")
            del_btn = gr.Button("删除图片")

        def new_img():
            imgs = getImgs()
            #print(imgs)
            return imgs, imgs
        def get_select_index(evt: gr.SelectData):
            return evt.index

        gallery.select(get_select_index, None, selected)
        bigImg = gr.Image(label="大图")
        
        def del_img(imgs, index):
            index = int(index)
            os.remove(imgs[index])
            imgs.pop(index)
            return imgs, imgs
        def show_img(imgs, index):
            index = int(index)
            return imgs[index]
        del_btn.click(del_img, [imgs, selected], [imgs, gallery])
        show_btn.click(show_img, [imgs, selected], [bigImg])
        new_btn.click(new_img, None, [imgs, gallery])  
            
                    
        
            
#demo.launch()
demo.launch(server_name="0.0.0.0",server_port=7860 )