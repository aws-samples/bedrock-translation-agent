# Multi-Model agentic & reflective translation workflow in Aamzon Bedrock

> Inspired by [andrewyng/translation-agent](https://github.com/andrewyng/translation-agent)，enabling it run in [Amazon Bedrock](https://aws.amazon.com/bedrock/)

A reflection agentic workflow for machine translation. The main step are:
1. Prompt an LLM to translation a text from `source_lang` to `target_lang`
2. Have the LLM reflect on the translation to come up with constructive suggestions for improving it;
3. Use the suggestions to improve the translation.

## Key features
With the support of Amazon Bedrock for multiple models, different models can be used in different steps to fully leverage the advantages of LLM.

## Getting Started

Following below steps to get started with `bedrock-translation-agent`: 

### Installation: 

- Local AWS cerdentials are required. In Amazon EC2, instance profile is the best option; and if in your local environment, your might need to configure the [configuration file](https://docs.aws.amazon.com/cli/v1/userguide/cli-configure-files.html) or [environment variables](https://docs.aws.amazon.com/cli/v1/userguide/cli-configure-envvars.html).

- Python package manage tool [Poetry](https://python-poetry.org/) is required, you can follow [the installation guide](https://python-poetry.org/docs/#installation) to install it. Depending on your local environment, this might work: 
```shell
curl -sSL https://install.python-poetry.org | python3 -
```

- After checking out the code, use Poetry to install dependencies:

```shell
poetry install
```

### Usage:

- Using default model:
```python
from bedrock_translation_agent.libs.translation import Translation

translation = Translation(
    source_text="Translation text here.",
    source_lang="English",
    target_lang="Chinese",
    country="China"
)

print(translation.translate())
```

- Using different models for each step:

>  The models supported in Amazon Bedrock are defined in`libs/bedrock_models.py`, and you can use the method `Translation.set_models` to declare the model for each step of translation.

**Note**: The code uses system_prompt and user_prompt. If the model you specified does not support system prompt, you need to modify the API parameters in `libs/bedrock.py`

```python
from bedrock_translation_agent.libs.bedrock_model import BedrockModel
from bedrock_translation_agent.libs.translation import Translation

translation = Translation(
    source_text=long_text,
    source_lang="English",
    target_lang="Chinese",
    country="China"
).set_models(
    init_model=BedrockModel.CLAUDE_3_SONNET_1_0,
    reflect_on_model=BedrockModel.CLAUDE_3_SONNET_1_0,
    improve_model=BedrockModel.CLAUDE_3_SONNET_1_0,
)

print(translation.translate())
```

## Related work

A few academic research groups are also starting to look at LLM-based and agentic translation. We think it’s early days for this field!
- *ChatGPT MT: Competitive for High- (but not Low-) Resource Languages*, Robinson et al. (2023), https://arxiv.org/pdf/2309.07423
- *How to Design Translation Prompts for ChatGPT: An Empirical Study*, Gao et al. (2023), https://arxiv.org/pdf/2304.02182v2
- *Beyond Human Translation: Harnessing Multi-Agent Collaboration for Translating Ultra-Long Literary Texts*, Wu et al. (2024),  https://arxiv.org/pdf/2405.11804


## License

This library is licensed under the MIT-0 License. See the LICENSE file.
