
from bedrock_translation_agent.libs.bedrock import Bedrock
from bedrock_translation_agent.libs.bedrock_model import BedrockModel


class OneChunkTranslation:

    _init_translation_model = BedrockModel.CLAUDE_3_SONNET_1_0
    _reflect_on_translation_model = BedrockModel.CLAUDE_3_SONNET_1_0
    _improve_translation_model = BedrockModel.CLAUDE_3_SONNET_1_0

    def __init__(self, source_text: str, source_lang: str, target_lang: str, country: str = ""):
        self.__bedrock = Bedrock()
        self.__source_text = source_text
        self.__source_lang = source_lang
        self.__target_lang = target_lang
        self.__country = country

    def do(self):
        pre_translation = self.__init_translation()
        reflection = self.__reflect_on_translation(pre_translation)
        translation = self.__improve_translation(
            pre_translation=pre_translation,
            reflection=reflection
        )

        return translation

    def __init_translation(self) -> str:
        sys_message_tpl = "You are an expert linguist, specializing in translation from {source_lang} to {target_lang}."
        system_prompt = sys_message_tpl.format(
            source_lang=self.__source_lang,
            target_lang=self.__target_lang
        )

        translate_message_tpl = """This is an {source_lang} to {target_lang} translation , please provide the {target_lang} translate from this text. \
Do not provide any explanations or text apart from the translation.
{source_lang}: {source_text}

{target_lang}:"""
        translate_prompt = translate_message_tpl.format(
            source_lang=self.__source_lang,
            target_lang=self.__target_lang,
            source_text=self.__source_text
        )

        return self.__bedrock.invoke_model(
            prompt=translate_prompt,
            system_msg=system_prompt,
            model=self._get_init_translation_model()
        )

    def __reflect_on_translation(self, pre_translation: str) -> str:
        sys_prompt_tpl = "You are an expert linguist specializing in translation from {source_lang} to {target_lang}. \
You will be provided with a source text and its translation and your goal is to improve the translation."
        sys_prompt = sys_prompt_tpl.format(
            source_lang=self.__source_lang,
            target_lang=self.__target_lang
        )

        reflection_prompt = ""
        if self.__country != "":
            reflection_prompt_tpl = """Your task is to carefully read a source text and a translation from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions to improve the translation. \
The final style and tone of the translation should match the style of {target_lang} colloquially spoken in {country}.

The source text and initial translation, delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT> and <TRANSLATION></TRANSLATION>, are as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's \n\
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),\n\
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
(iii) style (by ensuring the translations reflect the style of the source text and takes into account any cultural context),\n\
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_lang}).\n\

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else."""

            reflection_prompt = reflection_prompt_tpl.format(
                country=self.__country,
                source_lang=self.__source_lang,
                source_text=self.__source_text,
                target_lang=self.__target_lang,
                translation_1=pre_translation
            )
        else:
            reflection_prompt_tpl = """Your task is to carefully read a source text and a translation from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions to improve the translation. \

The source text and initial translation, delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT> and <TRANSLATION></TRANSLATION>, are as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's \n\
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),\n\
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
(iii) style (by ensuring the translations reflect the style of the source text and takes into account any cultural context),\n\
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_lang}).\n\

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else."""

            reflection_prompt = reflection_prompt_tpl.format(
                source_lang=self.__source_lang,
                target_lang=self.__target_lang,
                source_text=self.__source_text,
                translation_1=pre_translation
            )

        return self.__bedrock.invoke_model(
            prompt=reflection_prompt,
            system_msg=sys_prompt,
            model=self._get_reflect_on_translation_model()
        )

    def __improve_translation(self, pre_translation: str, reflection: str) -> str:
        sys_prompt_tpl = "You are an expert linguist, specializing in translation editing from {source_lang} to {target_lang}."
        sys_prompt = sys_prompt_tpl.format(
            source_lang=self.__source_lang,
            target_lang=self.__target_lang
        )

        improve_prompt_tpl = """Your task is to carefully read, then edit, a translation from {source_lang} to {target_lang}, taking into
account a list of expert suggestions and constructive criticisms.

The source text, the initial translation, and the expert linguist suggestions are delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT>, <TRANSLATION></TRANSLATION> and <EXPERT_SUGGESTIONS></EXPERT_SUGGESTIONS> \
as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

<EXPERT_SUGGESTIONS>
{reflection}
</EXPERT_SUGGESTIONS>

Please take into account the expert suggestions when editing the translation. Edit the translation by ensuring:

(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules and ensuring there are no unnecessary repetitions), \
(iii) style (by ensuring the translations reflect the style of the source text)
(iv) terminology (inappropriate for context, inconsistent use), or
(v) other errors.

Output only the new translation and nothing else."""
        improve_prompt = improve_prompt_tpl.format(
            source_text=self.__source_text,
            source_lang=self.__source_lang,
            target_lang=self.__target_lang,
            translation_1=pre_translation,
            reflection=reflection
        )

        return self.__bedrock.invoke_model(
            prompt=improve_prompt,
            system_msg=sys_prompt,
            model=self._get_improve_translation_model()
        )

    def set_init_model(self, model: BedrockModel):
        self._init_translation_model = model
        return self

    def set_reflect_on_model(self, model: BedrockModel):
        self._reflect_on_translation_model = model
        return self

    def set_improve_model(self, model: BedrockModel):
        self._improve_translation_model = model
        return self

    def _get_init_translation_model(self) -> BedrockModel:
        return self._init_translation_model

    def _get_reflect_on_translation_model(self) -> BedrockModel:
        return self._reflect_on_translation_model

    def _get_improve_translation_model(self) -> BedrockModel:
        return self._improve_translation_model
