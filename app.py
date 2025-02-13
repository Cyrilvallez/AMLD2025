
import queue
import copy
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import gradio as gr

device = "cuda:0"
model_id = "meta-llama/Llama-3.2-3B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    token="<ADD_TOKEN_HERE>",
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id, token="<ADD_TOKEN_HERE>")

system_prompt = f"You are a helpful assistant named Llama."


def text_generation(prompt:str, conversation: list[dict], max_new_tokens: int = 500, streamer = None):
    """Add a new turn to the conversation"""
    conversation.append(
        {"role": "user", "content": prompt}
    )
    inputs = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, return_tensors = "pt").to(device)
    mask = torch.ones_like(inputs)
    input_length = inputs.shape[-1]

    outputs = model.generate(inputs, attention_mask=mask, do_sample=True, temperature=0.8, top_k=50, max_new_tokens=max_new_tokens,
                             pad_token_id=tokenizer.eos_token_id, streamer=streamer)
    text_output = tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)[0]

    conversation.append(
        {"role": "assistant", "content": text_output}
    )

    return conversation


def text_continuation(conversation: list[dict], max_new_tokens: int = 500, streamer = None):
    """Continue the last answer in the conversation"""
    inputs = tokenizer.apply_chat_template(conversation, continue_final_message=True, return_tensors = "pt").to(device)
    mask = torch.ones_like(inputs)
    input_length = inputs.shape[-1]

    outputs = model.generate(inputs, attention_mask=mask, do_sample=True, temperature=0.8, top_k=50, max_new_tokens=max_new_tokens,
                             pad_token_id=tokenizer.eos_token_id, streamer=streamer)
    text_output = tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)[0]

    conversation[-1]["content"] = conversation[-1]["content"] + text_output

    return conversation


def retry_generation(conversation: list[dict], max_new_tokens: int = 500, streamer = None):
    """Remove and retry the last answer in the conversation"""
    prompt = conversation[-2]["content"]
    conversation = conversation[:-2]
    return text_generation(prompt, conversation, max_new_tokens, streamer)


def text_generation_streamed(prompt: str, conversation: list[dict]):
    """Same as `text_generation`, but yield tokens as soon as they are availabe"""
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, timeout=5, skip_special_tokens=True)
    conv_copy = copy.deepcopy(conversation)
    conv_copy.append(
        {"role": "user", "content": prompt}
    )
    conv_copy.append(
        {"role": "assistant", "content": ""}
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(text_generation, prompt, conversation, max_new_tokens=500, streamer=streamer)

        # Get results from the streamer and yield them
        try:
            for new_text in streamer:
                conv_copy[-1]["content"] = conv_copy[-1]["content"] + new_text
                yield "", conv_copy

        # If for some reason the queue (from the streamer) is still empty after timeout, we probably
        # encountered an exception
        except queue.Empty:
            e = future.exception()
            if e is not None:
                raise gr.Error(f'The following error happened during generation: {repr(e)}')
            else:
                raise gr.Error(f'Generation timed out (no new tokens were generated after 10s)')

        # # Get final result
        conv = future.result()
        yield "", conv


def text_continuation_streamed(conversation: list[dict]):
    """Same as `text_continuation`, but yield tokens as soon as they are availabe"""
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, timeout=5, skip_special_tokens=True)
    conv_copy = copy.deepcopy(conversation)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(text_continuation, conversation, max_new_tokens=500, streamer=streamer)

        # Get results from the streamer and yield them
        try:
            for new_text in streamer:
                conv_copy[-1]["content"] = conv_copy[-1]["content"] + new_text
                yield conv_copy

        # If for some reason the queue (from the streamer) is still empty after timeout, we probably
        # encountered an exception
        except queue.Empty:
            e = future.exception()
            if e is not None:
                raise gr.Error(f'The following error happened during generation: {repr(e)}')
            else:
                raise gr.Error(f'Generation timed out (no new tokens were generated after 10s)')

        # # Get final result
        conv = future.result()
        yield conv


def retry_generation_streamed(conversation: list[dict]):
    """Same as `retry_generation`, but yield tokens as soon as they are availabe"""
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, timeout=5, skip_special_tokens=True)
    conv_copy = copy.deepcopy(conversation)
    conv_copy[-1]["content"] = ""

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(retry_generation, conversation, max_new_tokens=500, streamer=streamer)

        # Get results from the streamer and yield them
        try:
            for new_text in streamer:
                conv_copy[-1]["content"] = conv_copy[-1]["content"] + new_text
                yield conv_copy

        # If for some reason the queue (from the streamer) is still empty after timeout, we probably
        # encountered an exception
        except queue.Empty:
            e = future.exception()
            if e is not None:
                raise gr.Error(f'The following error happened during generation: {repr(e)}')
            else:
                raise gr.Error(f'Generation timed out (no new tokens were generated after 10s)')

        # # Get final result
        conv = future.result()
        yield conv


def set_system_prompt(system_prompt: str, conversation: list[dict]):
    conversation[0]["content"] = system_prompt
    return conversation


# Just some urls for avatar images
hf_logo_url = "https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.png"
llama_logo_url = "https://storage.googleapis.com/pod_public/1300/169474.jpg"

# UI components
conversation = gr.Chatbot(value=[{"role": "system", "content": system_prompt}], label='Conversation', type="messages",
                          height=500, avatar_images=(hf_logo_url, llama_logo_url))
prompt = gr.Textbox(placeholder='Write your prompt here.', label='Prompt')
system_prompt = gr.Textbox(value=system_prompt, label='')
generate_button = gr.Button('▶️ Submit', variant='primary')
continue_button = gr.Button('🔂 Continue', variant='primary')
retry_button = gr.Button('🔄 Retry', variant='primary')
clear_button = gr.Button('🗑 Clear')
stop_button = gr.Button('🛑 Stop', variant='stop')
examples = ["Who are you?", "How cool are Hugging Face libraries? 🤗", "Write a Python function to get Fibonacci numbers", "What is the meaning of life?"]

# UI rendering and logic
with gr.Blocks() as demo:
    
    conversation.render()
    prompt.render()

    with gr.Row():
        generate_button.render()
        continue_button.render()
        retry_button.render()
        clear_button.render()
    stop_button.render()
    gr.Examples(examples, inputs=prompt)

    with gr.Accordion("System prompt", open=False):
        system_prompt.render()

    # Perform chat generation when clicking the button or pressing enter
    text_generation_event = gr.on(triggers=[generate_button.click, prompt.submit], fn=text_generation_streamed,
                                  inputs=[prompt, conversation], outputs=[prompt, conversation])
    
    text_continuation_event = gr.on(triggers=[continue_button.click], fn=text_continuation_streamed,
                                  inputs=conversation, outputs=conversation)

    retry_generation_event = gr.on(triggers=[retry_button.click], fn=retry_generation_streamed,
                                  inputs=conversation, outputs=conversation)

    stop_button.click(fn=None, inputs=None, outputs=None, cancels=[text_generation_event, text_continuation_event, retry_generation_event])

    clear_button.click(lambda conv: conv[:1], inputs=conversation, outputs=conversation)

    system_prompt.submit(set_system_prompt, inputs=[system_prompt, conversation], outputs=conversation)
    
demo.queue().launch(share=True)
