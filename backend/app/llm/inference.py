from backend.app.llm.load_model import load_model

model, tokenizer = load_model()


def generate_response(prompt):

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)