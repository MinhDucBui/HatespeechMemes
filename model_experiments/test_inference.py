from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch

model_path = "/Users/duc/Desktop/Projects/Ongoing/open_llm/models/llava_1_6_vicuna/models--liuhaotian--llava-v1.6-vicuna-13b/snapshots/22422b4c3a3ef1ba52aca074cc9021216877ce5d"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
# print(processor)
# das
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    #low_cpu_mem_usage=True
)
model.to(device)
