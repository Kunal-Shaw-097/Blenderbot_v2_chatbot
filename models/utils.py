from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer, AutoModelForCausalLM, AutoTokenizer
import torch
import logging

logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

class Load_Model:    
    def __init__(self, model_name):
        if model_name == 'Facebook_Blenderbot':
            self.model_id = "facebook/blenderbot-400M-distill"
            self.tokenizer = BlenderbotTokenizer.from_pretrained(self.model_id, truncation_side='left')
            self.model = BlenderbotForConditionalGeneration.from_pretrained(self.model_id)
        elif model_name == 'Microsoft_Dialogpt':
            self.model_id = "microsoft/DialoGPT-medium"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, truncation_side='left')
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        if torch.cuda.is_available():
            self.model.cuda()

class Generate:
    def __init__(self):
        self.History = []

    def generate(self, Model, prompt):
        if Model.model_id == 'facebook/blenderbot-400M-distill':
            if len(self.History) != 0 :
                past_conv = "".join(self.History)
                prompt = past_conv + "    " + prompt
            encoded_input = Model.tokenizer([prompt], return_tensors='pt', max_length = 128)
            if torch.cuda.is_available():
                encoded_input = encoded_input.to('cuda')
            output = Model.model.generate(**encoded_input, do_sample = True)
            answer = Model.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
            if len(self.History) == 0:
                self.History.append(prompt + '    ' + answer + '  ')
            else :
                self.History.append('  ' + prompt + '    ' + answer + '  ')
            return answer
        
        elif Model.model_id == "microsoft/DialoGPT-medium":
            new_user_input_ids = Model.tokenizer.encode(prompt + Model.tokenizer.eos_token , return_tensors='pt', max_length = 128)
            if len(self.History) == 0:
                bot_input_ids = new_user_input_ids
            else :
                bot_input_ids = torch.cat([self.History, new_user_input_ids], dim=-1)
                if bot_input_ids.shape[1] > 100 :
                    bot_input_ids = bot_input_ids.flip(-1)[:, :100].flip(-1)
            if torch.cuda.is_available():
                bot_input_ids = bot_input_ids.to('cuda')
            chat_history_ids = Model.model.generate(bot_input_ids, max_length=200,
                                                    pad_token_id=Model.tokenizer.eos_token_id,  
                                                    no_repeat_ngram_size=3,       
                                                    do_sample=True, 
                                                    top_k=100, 
                                                    top_p=0.7,
                                                    temperature=0.4)
            self.History = chat_history_ids.cpu()
            answer = Model.tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
            return answer
    
def model_info(model_name):
    if model_name == 'Facebook_Blenderbot':
        return "<ul><li>Parameters - 400 Millions.</li><li>Context_length - 128 tokens(Chatgpt has 4096 tokens).</li>" + \
                "<li>Link for more <a href= https://huggingface.co/docs/transformers/model_doc/blenderbot> info</a>.</li></ul>"
    else:
        return "<ul><li>Parameters - 345 Millions.</li><li>Context_length - 128 tokens(Chatgpt has 4096 tokens).</li>" + \
        "<li>Link for more <a href= https://huggingface.co/docs/transformers/model_doc/dialogpt> info</a>.</li></ul>"
        
        