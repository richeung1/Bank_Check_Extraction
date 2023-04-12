# Installling dependencies
from PIL import Image
import torch
from transformers import TrOCRProcessor
from transformers import VisionEncoderDecoderModel


# Loading Microsoft TrOCR Large Handwritten model. Finetuned on IAM Dataset
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
model.to(device)


# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id

# make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size

# set beam search parameters
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 64
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4


# Function to take in an image and return generated text
def process_image(image):
    # prepare image
    pixel_values = processor(image, return_tensors="pt").pixel_values

    # generate (no beam search)
    generated_ids = model.generate(pixel_values.to(device))

    # decode
    generated_ids_tensor = generated_ids[0]
    generated_text = processor.decode(generated_ids_tensor, skip_special_tokens=True)

    return generated_text


image = Image.open(r'C:\Users\Ricky\Desktop\Capstone-Project-OCR Part\Images\handwriting.JPG')
image.show()

print(process_image(image))