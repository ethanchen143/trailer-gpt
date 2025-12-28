import gradio as gr

from server import generate_text, list_checkpoints


def available_checkpoints():
    ckpts = list_checkpoints()
    return [str(c) for c in ckpts]


def infer(prompt, checkpoint, max_new_tokens, temperature, top_k, top_p):
    return generate_text(
        prompt=prompt,
        checkpoint=checkpoint if checkpoint else None,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_k=int(top_k) if top_k else None,
        top_p=float(top_p) if top_p else None,
    )


def build_ui():
    ckpts = available_checkpoints()
    default_ckpt = ckpts[-1] if ckpts else ""

    with gr.Blocks() as demo:
        gr.Markdown("## trailer_gpt")

        with gr.Row():
            checkpoint_dd = gr.Dropdown(
                label="Checkpoint",
                choices=ckpts,
                value=default_ckpt,
                allow_custom_value=True,
            )
            refresh_btn = gr.Button("Refresh checkpoints")

        prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Enter prompt...")
        with gr.Row():
            max_new_tokens = gr.Slider(10, 400, value=120, step=10, label="max_new_tokens")
            temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="temperature")
        with gr.Row():
            top_k = gr.Slider(0, 200, value=50, step=5, label="top_k (0=off)")
            top_p = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="top_p (0=off)")

        run_btn = gr.Button("Generate")
        output = gr.Textbox(label="Output", lines=10)

        def on_refresh():
            ck = available_checkpoints()
            return gr.Dropdown(choices=ck, value=ck[-1] if ck else "")

        refresh_btn.click(fn=on_refresh, outputs=checkpoint_dd)
        run_btn.click(
            fn=infer,
            inputs=[prompt, checkpoint_dd, max_new_tokens, temperature, top_k, top_p],
            outputs=output,
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860)