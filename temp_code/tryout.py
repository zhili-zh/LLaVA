from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import torch
import argparse

print(torch.cuda.is_available())

model_path = '../../../LLM_models/llava-v1.6-vicuna-7b/'

# tokenizer, model, image_processor, context_len = load_pretrained_model(
#     model_path=model_path,
#     model_base=None,
#     model_name=get_model_name_from_path(model_path)
# )

# model_path = "../../../LLM_models/llava-v1.6-vicuna-7b/"
# prompt = "Can you describe this image?"
# image_file = "./imgs/wall_mt.jpg"

# python tryout.py --image-file "./imgs/wall_mt.jpg" --query "Can you describe this image?"
# python tryout.py --image-file "./imgs/2_109.png" --query "Can you describe this image if i tell you this is a birdeye view of an intersection?"
# python tryout.py --image-file "./imgs/1_119.png" \
# --query "This image is a birdeye view of an intersection. You can see three green vehicles and two red vehicles in the picture"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=model_path)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)