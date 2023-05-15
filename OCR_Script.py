# Installling dependencies
import torch
from transformers import TrOCRProcessor
from transformers import VisionEncoderDecoderModel


class loadmodel:

    # Function to take in an image and return generated text
    def __init__(self):
        # Loading Microsoft TrOCR Large Handwritten model. Finetuned on IAM Dataset
        torch.cuda.empty_cache()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
        self.model.load_state_dict(torch.load('model\TrOCR_Bank_Check_Extraction_0.21221186534833886.pth'))

        self.model.to(self.device)

        # set special tokens used for creating the decoder_input_ids from the labels
        self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id

        # make sure vocab size is set correctly
        self.model.config.vocab_size = self.model.config.decoder.vocab_size

        # set beam search parameters
        self.model.config.eos_token_id = self.processor.tokenizer.sep_token_id
        self.model.config.max_length = 64
        self.model.config.early_stopping = True
        self.model.config.no_repeat_ngram_size = 3
        self.model.config.length_penalty = 2.0
        self.model.config.num_beams = 4


class OCR:
    # Inherit loadmodel class
    def __init__(self, loadmodel):
        self.loadmodel = loadmodel

    def process_image(self, image):
        # prepare image
        pixel_values = self.loadmodel.processor(image, return_tensors="pt").pixel_values

        # generate (no beam search)
        generated_ids = self.loadmodel.model.generate(pixel_values.to(self.loadmodel.device))

        # decode
        generated_ids_tensor = generated_ids[0]
        generated_text = self.loadmodel.processor.decode(generated_ids_tensor, skip_special_tokens=True)

        return generated_text