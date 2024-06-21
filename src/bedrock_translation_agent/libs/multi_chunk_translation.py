from typing import List

from bedrock_translation_agent.libs.bedrock import Bedrock
from bedrock_translation_agent.libs.bedrock_model import BedrockModel
from bedrock_translation_agent.libs.utils import split_text_by_token_size


class MultiChunkTranslation:

    _init_translation_model = BedrockModel.CLAUDE_3_SONNET_1_0
    _reflect_on_translation_model = BedrockModel.CLAUDE_3_SONNET_1_0
    _improve_translation_model = BedrockModel.CLAUDE_3_SONNET_1_0

    def __init__(self, source_text: str, source_lang: str, target_lang: str, country: str = ""):
        self.__bedrock = Bedrock()
        self.__source_text_chunks = split_text_by_token_size(source_text)
        self.__source_lang = source_lang
        self.__target_lang = target_lang,
        self.__country = country

    def do(self):
        pre_translation_chunks = self.__init_translation()
        reflection_chunks = self.__reflect_on_translation(
            pre_translation_chunks=pre_translation_chunks
        )
        improve_translation = self.__improve_translation(
            pre_translation_chunks=pre_translation_chunks, reflection_chunks=reflection_chunks
        )

        return "".join(improve_translation)

    def __init_translation(self) -> List[str]:
        sys_prompt = "You are an expert linguist, specializing in translation from {source_lang} to {target_lang}.".format(
            source_lang=self.__source_lang, target_lang=self.__target_lang
        )

        translation_prompt_tpl = """Your task is provide a professional translation from {source_lang} to {target_lang} of PART of a text.

The source text is below, delimited by XML tags <SOURCE_TEXT> and </SOURCE_TEXT>. Translate only the part within the source text
delimited by <TRANSLATE_THIS> and </TRANSLATE_THIS>. You can use the rest of the source text as context, but do not translate any
of the other text. Do not output anything other than the translation of the indicated part of the text.

<SOURCE_TEXT>
{tagged_text}
</SOURCE_TEXT>

To reiterate, you should translate only this part of the text, shown here again between <TRANSLATE_THIS> and </TRANSLATE_THIS>:
<TRANSLATE_THIS>
{chunk_to_translate}
</TRANSLATE_THIS>

Output only the translation of the portion you are asked to translate, and nothing else.
"""
        translation_chunks = []
        for i in range(len(self.__source_text_chunks)):
            tagged_text = (
                "".join(self.__source_text_chunks[0:i])
                + "<TRANSLATE_THIS>"
                + self.__source_text_chunks[i]
                + "</TRANSLATE_THIS>"
                + "".join(self.__source_text_chunks[i + 1:])
            )

            prompt = translation_prompt_tpl.format(
                source_lang=self.__source_lang, target_lang=self.__target_lang,
                tagged_text=tagged_text, chunk_to_translate=self.__source_text_chunks[i]
            )

            translation = self.__bedrock.invoke_model(
                prompt=prompt,
                system_msg=sys_prompt,
                model=self._get_init_translation_model()
            )
            translation_chunks.append(translation)

        return translation_chunks

    def __reflect_on_translation(self, pre_translation_chunks: List[str]) -> List[str]:
        sys_prompt = "You are an expert linguist specializing in translation from {source_lang} to {target_lang}. \
You will be provided with a source text and its translation and your goal is to improve the translation.".format(
            source_lang=self.__source_lang, target_lang=self.__target_lang
        )

        reflection_prompt_tpl = ""
        if self.__country != "":
            reflection_prompt_tpl = """Your task is to carefully read a source text and part of a translation of that text from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions for improving the translation.
The final style and tone of the translation should match the style of {target_lang} colloquially spoken in {country}.

The source text is below, delimited by XML tags <SOURCE_TEXT> and </SOURCE_TEXT>, and the part that has been translated
is delimited by <TRANSLATE_THIS> and </TRANSLATE_THIS> within the source text. You can use the rest of the source text
as context for critiquing the translated part.

<SOURCE_TEXT>
{tagged_text}
</SOURCE_TEXT>

To reiterate, only part of the text is being translated, shown here again between <TRANSLATE_THIS> and </TRANSLATE_THIS>:
<TRANSLATE_THIS>
{chunk_to_translate}
</TRANSLATE_THIS>

The translation of the indicated part, delimited below by <TRANSLATION> and </TRANSLATION>, is as follows:
<TRANSLATION>
{translation_1_chunk}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's:\n\
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),\n\
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
(iii) style (by ensuring the translations reflect the style of the source text and takes into account any cultural context),\n\
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_lang}).\n\

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else."""
        else:
            reflection_prompt_tpl = """Your task is to carefully read a source text and part of a translation of that text from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions for improving the translation.

The source text is below, delimited by XML tags <SOURCE_TEXT> and </SOURCE_TEXT>, and the part that has been translated
is delimited by <TRANSLATE_THIS> and </TRANSLATE_THIS> within the source text. You can use the rest of the source text
as context for critiquing the translated part.

<SOURCE_TEXT>
{tagged_text}
</SOURCE_TEXT>

To reiterate, only part of the text is being translated, shown here again between <TRANSLATE_THIS> and </TRANSLATE_THIS>:
<TRANSLATE_THIS>
{chunk_to_translate}
</TRANSLATE_THIS>

The translation of the indicated part, delimited below by <TRANSLATION> and </TRANSLATION>, is as follows:
<TRANSLATION>
{translation_1_chunk}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's:\n\
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),\n\
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
(iii) style (by ensuring the translations reflect the style of the source text and takes into account any cultural context),\n\
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_lang}).\n\

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else."""

        reflection_chunks = []
        for i in range(len(self.__source_text_chunks)):
            tagged_text = (
                "".join(self.__source_text_chunks[0:i])
                + "<TRANSLATE_THIS>"
                + self.__source_text_chunks[i]
                + "</TRANSLATE_THIS>"
                + "".join(self.__source_text_chunks[i + 1:])
            )

            if self.__country != "":
                prompt = reflection_prompt_tpl.format(
                    source_lang=self.__source_lang, target_lang=self.__target_lang, tagged_text=tagged_text,
                    chunk_to_translate=self.__source_text_chunks[
                        i], translation_1_chunk=pre_translation_chunks[i],
                    country=self.__country
                )

            else:
                prompt = reflection_prompt_tpl.format(
                    source_lang=self.__source_lang, target_lang=self.__target_lang, tagged_text=tagged_text,
                    chunk_to_translate=self.__source_text_chunks[
                        i], translation_1_chunk=pre_translation_chunks[i]
                )

            reflection = self.__bedrock.invoke_model(
                prompt=prompt,
                system_msg=sys_prompt,
                model=self._get_reflect_on_translation_model()
            )
            reflection_chunks.append(reflection)

        return reflection_chunks

    def __improve_translation(self, pre_translation_chunks: List[str], reflection_chunks: List[str]) -> List[str]:
        sys_prompt = "You are an expert linguist, specializing in translation editing from {source_lang} to {target_lang}.".format(
            source_lang=self.__source_lang, target_lang=self.__target_lang
        )

        improvement_prompt_tpl = """Your task is to carefully read, then improve, a translation from {source_lang} to {target_lang}, taking into
account a set of expert suggestions and constructive critisms. Below, the source text, initial translation, and expert suggestions are provided.

The source text is below, delimited by XML tags <SOURCE_TEXT> and </SOURCE_TEXT>, and the part that has been translated
is delimited by <TRANSLATE_THIS> and </TRANSLATE_THIS> within the source text. You can use the rest of the source text
as context, but need to provide a translation only of the part indicated by <TRANSLATE_THIS> and </TRANSLATE_THIS>.

<SOURCE_TEXT>
{tagged_text}
</SOURCE_TEXT>

To reiterate, only part of the text is being translated, shown here again between <TRANSLATE_THIS> and </TRANSLATE_THIS>:
<TRANSLATE_THIS>
{chunk_to_translate}
</TRANSLATE_THIS>

The translation of the indicated part, delimited below by <TRANSLATION> and </TRANSLATION>, is as follows:
<TRANSLATION>
{translation_1_chunk}
</TRANSLATION>

The expert translations of the indicated part, delimited below by <EXPERT_SUGGESTIONS> and </EXPERT_SUGGESTIONS>, is as follows:
<EXPERT_SUGGESTIONS>
{reflection_chunk}
</EXPERT_SUGGESTIONS>

Taking into account the expert suggestions rewrite the translation to improve it, paying attention
to whether there are ways to improve the translation's

(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules and ensuring there are no unnecessary repetitions), \
(iii) style (by ensuring the translations reflect the style of the source text)
(iv) terminology (inappropriate for context, inconsistent use), or
(v) other errors.
(vi) remove all XML tags in the output.

Output only the new translation of the indicated part and nothing else."""

        improvement_translation_chunks = []
        for i in range(len(self.__source_text_chunks)):
            tagged_text = (
                "".join(self.__source_text_chunks[0:i])
                + "<TRANSLATE_THIS>"
                + self.__source_text_chunks[i]
                + "</TRANSLATE_THIS>"
                + "".join(self.__source_text_chunks[i + 1:])
            )

            improvement_prompt = improvement_prompt_tpl.format(
                source_lang=self.__source_lang, target_lang=self.__target_lang, tagged_text=tagged_text,
                chunk_to_translate=self.__source_text_chunks[i],
                translation_1_chunk=pre_translation_chunks[i],
                reflection_chunk=reflection_chunks[i],
            )

            improvement_translation = self.__bedrock.invoke_model(
                system_msg=sys_prompt,
                prompt=improvement_prompt,
                model=self._get_improve_translation_model()
            )

            improvement_translation_chunks.append(improvement_translation)

        return improvement_translation_chunks

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
