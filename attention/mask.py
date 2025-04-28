import sys
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

from PIL import Image, ImageDraw, ImageFont
from transformers import TFBertForMaskedLM, AutoTokenizer

MODEL_NAME = "bert-base-uncased"
TOP_K_PREDICTIONS = 3

FONT_PATH = "assets/fonts/OpenSans-Regular.ttf"
FONT_SIZE = 28
GRID_SPACING = 40
TOKEN_PADDING = 200
FONT = ImageFont.truetype(FONT_PATH, FONT_SIZE)


def main():
    user_input = input("Enter text: ")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoded = tokenizer(user_input, return_tensors="tf")
    mask_pos = find_mask_position(tokenizer.mask_token_id, encoded)

    if mask_pos is None:
        sys.exit(f"Please include a mask token {tokenizer.mask_token} in your input.")

    model = TFBertForMaskedLM.from_pretrained(MODEL_NAME)
    output = model(**encoded, output_attentions=True)

    mask_logits = output.logits[0, mask_pos]
    top_indices = tf.math.top_k(mask_logits, k=TOP_K_PREDICTIONS).indices.numpy()

    for idx in top_indices:
        prediction = tokenizer.decode([idx])
        print(user_input.replace(tokenizer.mask_token, prediction))

    create_attention_maps(encoded.tokens(), output.attentions)


def find_mask_position(mask_id, encoded_input):
    """
    Locate the index of the [MASK] token in the encoded input.
    """
    for index, token_id in enumerate(encoded_input.input_ids[0]):
        if token_id == mask_id:
            return index
    return None


def attention_to_color(score):
    """
    Convert an attention score into a grayscale color value.
    """
    score = score.numpy()
    grayscale_value = round(score * 255)
    return (grayscale_value, grayscale_value, grayscale_value)


def create_attention_maps(tokens, attention_layers):
    """
    Generate attention diagrams for each attention head in each layer.
    """
    for layer_idx, layer in enumerate(attention_layers):
        for head_idx in range(len(layer[0])):
            save_attention_image(
                layer_idx + 1,
                head_idx + 1,
                tokens,
                layer[0][head_idx]
            )


def save_attention_image(layer_num, head_num, tokens, attention_matrix):
    """
    Create and save an attention visualization.
    """
    canvas_size = GRID_SPACING * len(tokens) + TOKEN_PADDING
    canvas = Image.new("RGBA", (canvas_size, canvas_size), "black")
    draw = ImageDraw.Draw(canvas)

    for idx, token in enumerate(tokens):
        temp_img = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
        temp_draw = ImageDraw.Draw(temp_img)
        temp_draw.text(
            (canvas_size - TOKEN_PADDING, TOKEN_PADDING + idx * GRID_SPACING),
            token,
            fill="white",
            font=FONT
        )
        rotated = temp_img.rotate(90)
        canvas.paste(rotated, mask=rotated)

        _, _, w, _ = draw.textbbox((0, 0), token, font=FONT)
        draw.text(
            (TOKEN_PADDING - w, TOKEN_PADDING + idx * GRID_SPACING),
            token,
            fill="white",
            font=FONT
        )

    for row in range(len(tokens)):
        y = TOKEN_PADDING + row * GRID_SPACING
        for col in range(len(tokens)):
            x = TOKEN_PADDING + col * GRID_SPACING
            color = attention_to_color(attention_matrix[row][col])
            draw.rectangle((x, y, x + GRID_SPACING, y + GRID_SPACING), fill=color)

    filename = f"attention_layer{layer_num}_head{head_num}.png"
    canvas.save(filename)


if __name__ == "__main__":
    main()
