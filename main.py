import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
import tempfile

from dotenv import load_dotenv
load_dotenv()

#テンプレートの用意
template_qa = """
        あなたは親切なアシスタントです。下記の質問に日本語で回答してください。
        質問：{question}
        回答：
        """

prompt_qa = PromptTemplate(
            input_variables=["question"],
            template=template_qa,
        )

prompt_template_smr = """
回答は日本語で出力してください。
以下の内容を簡潔に要約てください。
内容:{text}
回答：
"""
prompt_smr = PromptTemplate(
    template=prompt_template_smr, 
    input_variables=["text"]
    )

def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        return temp_file.name   


st.title("論文Q&A（英語も可）")
uploaded_file = st.file_uploader("PDFファイルを選択してください", type="pdf")

if uploaded_file is not None:
    # ファイルを処理するために、一時的に保存されたPDFファイルのパスを使用
    # （例：pdf_text = read_pdf(temp_file_path)）
    temp_file_path = save_uploaded_file(uploaded_file)

    #PDFの読み込み
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()

    #埋め込み型の作成
    text_splitter_qa = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts_qa = text_splitter_qa.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(texts_qa, embeddings)

    #モデルとレトリーバーの作成
    llm_qa=ChatOpenAI(model_name="gpt-3.5-turbo")
    retriever = vectordb.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm_qa, chain_type="stuff", retriever=retriever)
   
    # 処理が終わったら、一時ファイルを削除
    os.unlink(temp_file_path)

    # ラジオボタンで「要約」や「質問」の選択肢を作成
    option = st.radio("実行したい処理を選択してください", ("要約", "質問"))

    # 実行ボタンを作成
    if option == "要約":
        st.write("要約を実行する")
        st.write("時間がかかります……")
        if st.button("実行"):
            llm_smr = OpenAI(model_name="text-davinci-003", temperature=0)
            text_splitter_smr = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=20)
            texts_smr = text_splitter_smr.split_documents(documents)
            chain = load_summarize_chain(
                            llm=llm_smr, 
                            chain_type="map_reduce",
                            return_intermediate_steps=True,
                            map_prompt=prompt_smr, 
                            combine_prompt=prompt_smr
                            )
            
            sammary = chain(texts_smr, return_only_outputs=True)
            intermediate_steps_list = sammary['intermediate_steps']
            for sentence in intermediate_steps_list:
                st.write(sentence)

    elif option == "質問":
        st.title("質問内容")
        with st.form("question_form"):
            user_input = st.text_input("質問したい内容を記載してください:")

            if st.form_submit_button("送信"):
                st.write(f"入力されたテキスト：{user_input}")

                #プロンプトの作成
                prompt_text = prompt_qa.format(question=user_input)
                query = prompt_text

                #回答の表示
                anser = qa.run(query)
                st.write(anser)
            else:
                st.write("テキストボックスに入力して、送信ボタンを押してください。")
            
        
