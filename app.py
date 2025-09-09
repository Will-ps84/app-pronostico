import gradio as gr

def greet(name):
    return f"Hola {name}, tu app en Hugging Face está funcionando! 🚀"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch()
