from openai import OpenAI
import os
import sys



def ask_deepseek(question):
    client = OpenAI(
        base_url="https://api-inference.modelscope.cn/v1/",
        api_key="038a524d-63a6-4adf-a220-36cdc2e8d9be", #ModelScope Token
    )

    try:
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1",
            messages=[
                {"role": "system", "content": "you are a helpful assistant."},
                {"role": "user", "content": question},
            ],
            stream=True,
        )

        reasoning_content = ""
        answer_content = ""
        done_reasoning = False

        for chunk in response:
            delta = chunk.choices[0].delta

            reasoning_chunk = getattr(delta, "reasoning_content", None)
            answer_chunk = getattr(delta, "content", None)

            if reasoning_chunk:
                print(reasoning_chunk, end="", flush=True)
                reasoning_content += reasoning_chunk
            elif answer_chunk:
                if not done_reasoning:
                    print("\n\n === Final answer ===\n")
                    done_reasoning = True
                print(answer_chunk, end="", flush=True)
                answer_content += answer_chunk

    except Exception as e:
        print(f"An error occurred: {e}")

    return answer_content


if __name__ == "__main__":
    while True:
        question = input("\nYou: ")
        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        print("\nDeepSeek: ", end="")
        ask_deepseek(question)
