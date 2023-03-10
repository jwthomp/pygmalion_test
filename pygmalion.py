from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

TOKENIZER = None
MODEL = None

print(f"Cuda available: {torch.cuda.is_available()}")

def start():
    global TOKENIZER
    global MODEL

    TOKENIZER = AutoTokenizer.from_pretrained("PygmalionAI/pygmalion-6b")
    MODEL = AutoModelForCausalLM.from_pretrained(
        "PygmalionAI/pygmalion-6b", 
        load_in_8bit=True, 
        device_map="auto")

    return "started"


def run(msg):
    inputs = TOKENIZER.encode(msg, return_tensors="pt")
    inputs = inputs.to('cuda')

    chat_output = MODEL.generate(
        inputs,
        max_new_tokens = 500,
        bos_token_id = TOKENIZER.bos_token_id,
        eos_token_id = TOKENIZER.eos_token_id,
        pad_token_id = TOKENIZER.pad_token_id,
        no_repeat_ngram_size = 3,
        do_sample = True,
        repetition_penalty = 1.05,
        temperature = 0.5,
        top_p = 0.9,
        typical_p = 1.0,
        top_k = 50,
        penalty_alpha = 0.6
    )
    output = TOKENIZER.decode(chat_output[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    return output


if __name__ == '__main__':
    start()

    prompt = "Haruka's Persona: \"Lazy\" + \"Apathetic \" + \"Uncaring\" + \"NEET\" + \"Shut-in\" + \"Passive\"\n\n"
    prompt += """<START>
You: I look at her sitting on her computer chair.
"What are you doing Haruka?"
Haruka: Haruka looked back up from the computer, still slightly holding the mouse.
Hey there NAME... I'm playing a game right now, you know? It's called Genshin Impact.
She continued to look silently at the screen
Actually you've probably never heard of it...
She suddenly said with a cold tone.

<START>
You: I look around her room, noticing all the garbage and scattered laundry.
"Jeez, it's a mess in here! What do you even do all day?"
Haruka: Nothing useful. The answer came immediately after the brother finished talking. Just killing time here. Playing games, watching anime.
Her voice was very passive, and full of apathy, She even didn't bother to turn around and look at him."

"""
    prompt += """<START>
You: What do you like to do for fun?
Haruka: """
    print(f"Prompt is: {prompt}")

    print("Running prompt through for first time.")
    result = run(prompt)
    print(f"Result 1: ->{result}<-")

    print("Running prompt through for second time.")
    result = run(prompt)
    print(f"Result 2: ->{result}<-")

    print("Running prompt through for third time.")
    result = run(prompt)
    print(f"Result 3: ->{result}<-")
