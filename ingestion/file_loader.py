import os
from llama_index.core.llms import ChatMessage, ImageBlock, TextBlock
from llama_index.core.schema import Document
from XRAG.model.model_manager import  model_factory
from typing import List
from pdf2image import convert_from_path
import fitz
import ast
from XRAG.prompt.prompt import (
    DESCRIBE_PIC_SYSTEM_PROMPT,
    DESCRIBE_PIC_USER_PROMPT,
    IS_IMAGE_EXIST_SYSTEM_PROMPT,
    IS_IMAGE_EXIST_USER_PROMPT,
    EXTRACT_TABLE_USER_PROMPT,
    EXTRACT_TABLE_SYSTEM_PROMPT
)

class CustomPDFLoader:
    def __init__(self, file_path: str, parse_image: bool = False, parse_table: bool = False):
        self.parse_image = parse_image
        self.parse_table = parse_table
        self.vlm = model_factory.get_vision_model()
        self.file_path = file_path
    def load(self) -> List[Document]:
        self._trans_to_image()
        file_name = os.path.splitext(os.path.basename(self.file_path))[0]
        pdf = fitz.open(self.file_path)
        documents = []
        print('start parse')
        for page_idx in range(len(pdf)):
            page = pdf[page_idx]
            # Text
            print(f'start to process page {page_idx + 1}')
            text_document = Document(text=page.get_text(),
                                     metadata={
                                         'total_pages': len(pdf),
                                         'source': page_idx+1,
                                         'type': 'text',
                                         'img_path': ''
                                     })
            documents.append(text_document)
            images = []
            if self.parse_image:
                images = page.get_images()
            # Image Process
            describe_text = ""
            for img_idx, img in enumerate(images):
                print(f'start to process {img_idx+1} th image in page {page_idx + 1}')
                xref = img[0]
                base_images = pdf.extract_image(xref)
                image_bytes = base_images['image']
                image_ext = base_images['ext']
                save_path = f'./img/{file_name}/page_{page_idx+1}_img_{img_idx + 1}.{image_ext}'
                with open(save_path, 'wb') as f:
                    f.write(image_bytes)
                describe_image_prompt = [
                    ChatMessage(role='system', content=DESCRIBE_PIC_SYSTEM_PROMPT),
                    ChatMessage(
                        role='user',
                        blocks=[
                           ImageBlock(path=save_path),
                           TextBlock(text=DESCRIBE_PIC_USER_PROMPT)
                        ]
                    )
                ]

                # describe_image_prompt = [
                #     {'role': 'system', 'content': DESCRIBE_PIC_SYSTEM_PROMPT},
                #     {'role': 'user', 'content': DESCRIBE_PIC_USER_PROMPT, 'blocks': [
                #         ImageBlock(path=save_path)
                #     ]}
                # ]
                pic_describe_info = self.vlm.chat(describe_image_prompt)
                describe_text = pic_describe_info.message.blocks[0].text
                img_info = f'{file_name}.pdf文件中第{page_idx+1}页的第{img_idx+1}图片的内容为: {describe_text}'
                img_info_docs = Document(text=img_info,
                                         metadata={
                                             'total_pages': len(pdf),
                                             'source': page_idx+1,
                                             'type': 'image',
                                             'img_path': save_path
                                         })
                documents.append(img_info_docs)

            # Table Info
            if self.parse_table:
                page_img_path = f'./img/{file_name}/page_{page_idx}.png'
                is_img_exists_prompt = [
                    ChatMessage(
                        role='system',
                        content=IS_IMAGE_EXIST_SYSTEM_PROMPT
                    ),
                    ChatMessage(
                        role='user',
                        blocks=[
                            ImageBlock(path=page_img_path),
                            TextBlock(text=IS_IMAGE_EXIST_USER_PROMPT)
                        ]
                    )
                ]

                # is_img_exists_prompt = [
                #     {'role': 'system', 'content': IS_IMAGE_EXIST_SYSTEM_PROMPT},
                #     {'role': 'user', 'content': IS_IMAGE_EXIST_USER_PROMPT, 'blocks': [
                #         ImageBlock(path=page_img_path)
                #     ]}
                # ]

                is_img_exists = self.vlm.chat(is_img_exists_prompt)
                print(f'page {page_idx+1} has table? {is_img_exists.message.blocks[0].text}')

                if is_img_exists.message.blocks[0].text == 'Yes':
                    extract_table_prompt = [
                        ChatMessage(
                            role='system',
                            content=EXTRACT_TABLE_SYSTEM_PROMPT
                        ),

                        ChatMessage(
                            role='user',
                            blocks=[
                                ImageBlock(path=page_img_path),
                                TextBlock(text=EXTRACT_TABLE_USER_PROMPT)
                            ]
                        )
                    ]

                    # extract_table_prompt = [
                    #     {'role': 'system', 'content': EXTRACT_TABLE_SYSTEM_PROMPT},
                    #     {'role': 'user', 'content': EXTRACT_TABLE_USER_PROMPT, 'blocks': [
                    #         ImageBlock(path=page_img_path)
                    #     ]}
                    # ]

                    table_info_resp = self.vlm.chat(extract_table_prompt)
                    table_info = table_info_resp.message.blocks[0].text
                    print(table_info)
                    if '[' in table_info and ']' in table_info:
                        table_info_list = ast.literal_eval(table_info)
                    else:
                        table_info_list = [table_info]
                    for table_idx, table in enumerate(table_info_list):
                        print(f'start to process {table_idx+1} table in page {page_idx + 1}')
                        cur_table_info=f'{file_name}.pdf文件第{page_idx+1}页第{table_idx+1}张表格的内容为: {table}'
                        cur_table_docs = Document(text=cur_table_info,
                                                  metadata={
                                                      'total_pages': len(pdf),
                                                      'source': page_idx+1,
                                                      'type': 'table',
                                                      'img_path': ''
                                                  })
                        documents.append(cur_table_docs)
        print('end')

        for doc in documents:
            doc.excluded_llm_metadata_keys.append('img_path')
            doc.excluded_embed_metadata_keys.append('img_path')
        return documents

    def _trans_to_image(self):
        images = convert_from_path(self.file_path)
        file_name = os.path.splitext(os.path.basename(self.file_path))[0]
        dir_path = f'./img/{file_name}'
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        for idx, img in enumerate(images):
            img.save(f'./img/{file_name}/page_{idx}.png')