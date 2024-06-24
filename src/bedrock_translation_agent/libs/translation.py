from bedrock_translation_agent.libs.bedrock_model import BedrockModel
from bedrock_translation_agent.libs.multi_chunk_translation import \
    MultiChunkTranslation
from bedrock_translation_agent.libs.one_chunk_translation import \
    OneChunkTranslation
from bedrock_translation_agent.libs.utils import (MAX_TOKENS_PER_CHUNK,
                                                  num_tokens_in_string)


class Translation:

    __init_model: BedrockModel = BedrockModel.CLAUDE_3_SONNET_1_0
    __reflect_on_model: BedrockModel = BedrockModel.CLAUDE_3_SONNET_1_0
    __improve_model: BedrockModel = BedrockModel.CLAUDE_3_SONNET_1_0

    def __init__(
        self,
        source_lang: str,
        target_lang: str,
        source_text: str,
        country: str,
        max_tokens: int = MAX_TOKENS_PER_CHUNK
    ):
        self.__source_lang = source_lang
        self.__target_lang = target_lang
        self.__source_text = source_text
        self.__country = country
        self.__max_tokens = max_tokens

    def set_models(
        self,
        init_model: BedrockModel = BedrockModel.DEFAULT_MODEL,
        reflect_on_model: BedrockModel = BedrockModel.DEFAULT_MODEL,
        improve_model: BedrockModel = BedrockModel.DEFAULT_MODEL
    ):
        self.__init_model = init_model
        self.__reflect_on_model = reflect_on_model
        self.__improve_model = improve_model

        return self

    def translate(self):
        pass
        num_tokens_in_text = num_tokens_in_string(input_str=self.__source_text)
        if num_tokens_in_text < self.__max_tokens:
            return self._do_onechunk_translation()
        else:
            return self._do_multichunk_translation()

    def _do_onechunk_translation(self) -> str:
        translation = OneChunkTranslation(
            source_text=self.__source_text,
            source_lang=self.__source_lang,
            target_lang=self.__target_lang,
            country=self.__country
        )

        translation.set_init_model(self.__init_model)
        translation.set_reflect_on_model(self.__reflect_on_model)
        translation.set_improve_model(self.__improve_model)
        return translation.do()

    def _do_multichunk_translation(self) -> str:
        translation = MultiChunkTranslation(
            source_text=self.__source_text,
            source_lang=self.__source_lang,
            target_lang=self.__target_lang,
            country=self.__country
        )

        translation.set_init_model(self.__init_model)
        translation.set_reflect_on_model(self.__reflect_on_model)
        translation.set_improve_model(self.__improve_model)

        return translation.do()
