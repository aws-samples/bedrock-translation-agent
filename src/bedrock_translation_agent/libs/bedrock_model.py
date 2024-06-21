from enum import Enum, unique


@unique
class BedrockModel(Enum):
    '''
    Amazon Bedrock Model IDs, see: https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html#model-ids-arns
    '''

    # Amazon Titian
    AMAZON_TITAN_TEXT_G1_EXPRESS = "amazon.titan-text-express-v1"
    AMAZON_TITAN_TEXT_G1_LITE = "amazon.titan-text-lite-v1"
    AMAZON_TITAN_TEXT_PREMIER = "amazon.titan-text-premier-v1:0"

    # Anthropic Claude
    CLAUDE_2_0 = "anthropic.claude-v2"
    CLAUDE_2_1 = "anthropic.claude-v2:1"
    CLAUDE_INSTANT_1_X = "anthropic.claude-instant-v1"

    CLAUDE_3_HAIKU_1_0 = "anthropic.claude-3-haiku-20240307-v1:0"
    CLAUDE_3_SONNET_1_0 = "anthropic.claude-3-sonnet-20240229-v1:0"
    CLAUDE_3_OPUS_1_0 = "anthropic.claude-3-opus-20240229-v1:0"

    # AI21 Labs Jurassic
    AI21_J2_MID_1_X = "ai21.j2-mid-v1"
    AI21_J2_ULTRA_1_X = "ai21.j2-ultra-v1"

    # Cohere Command
    COHERE_COMMAND_14_X = "cohere.command-text-v14"
    COHERE_COMMAND_LIGHT_15_X = "cohere.command-light-text-v14"
    COHERE_COMMAND_R_1_X = "cohere.command-r-v1:0"
    COHERE_COMMAND_R_PLUS_1_X = "cohere.command-r-plus-v1:0"

    # Meta Llama
    MATA_LLAMA_2_CHAT_13B = "meta.llama2-13b-chat-v1"
    META_LLAMA_2_CHAT_70B = "meta.llama2-70b-chat-v1"
    META_LLAMA_3_8B_INSTRUCT = "meta.llama3-8b-instruct-v1:0"
    META_LLAMA_3_70B_INSTRUCT = "meta.llama3-70b-instruct-v1:0"

    # Mistral AI
    MISTRAL_7B_INSTRUCT = "mistral.mistral-7b-instruct-v0:2"
    MISTRAL_8X7B_INSTRUCT = "mistral.mixtral-8x7b-instruct-v0:1"
    MISTRAL_LARGE = "mistral.mistral-large-2402-v1:0"
    MISTRAL_SMALL = "mistral.mistral-small-2402-v1:0"
