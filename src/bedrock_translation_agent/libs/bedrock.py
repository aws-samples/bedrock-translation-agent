import boto3

from bedrock_translation_agent.libs.bedrock_model import BedrockModel


class Bedrock:

    def __init__(self, region="us-east-1") -> None:
        self._client = boto3.client("bedrock-runtime", region_name=region)
    
    #As the number of models supported by bedlock increases, users can update the model list in bedlock_model
    def invoke_model(
        self,
        prompt: str,
        system_msg: str = "You are a helpful assistant.",
        model: BedrockModel = BedrockModel.DEFAULT_MODEL,
        temperature: float = 0.3
    ) -> str:

        sys_prompt = [{"text": system_msg}]
        user_prompt = [{"role": "user", "content": [{"text": prompt}]}]
        response = self._client.converse(
            modelId=model.value,
            messages=user_prompt,
            system=sys_prompt,
            inferenceConfig={"temperature": temperature, "topP": 1}
        )

        return response['output']['message']['content'][0]['text']
