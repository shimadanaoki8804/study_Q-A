import streamlit as st
import os
from pdfminer.high_level import extract_text
from langchain.document_loaders import PyPDFLoader
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
import tempfile
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

#パス取得の関数
def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        return temp_file.name

#テキストを読み込む関数
def extract_text_from_pdf(file_path):
    text = extract_text(file_path)
    text = text.replace("A B S T R A C T", "Abstract")
    text = text.replace("ABSTRACT", "Abstract")
    text = text.replace("A B S T R A C T", "Abstract")
    text = text.replace("abstract", "Abstract")
    text = text.replace("a b s t r a c t", "Abstract")
    text = text.replace("INTRODUCTION", "Introduction")
    text = text.replace("I N T R O D U C T I O N", "Introduction")
    text = text.replace("introduction", "Introduction")
    text = text.replace("i n t r o d u c t i o n", "Introduction")

    # "Abstract" セクションのテキストを抽出
    abstract_start = text.find("Abstract")
    abstract_end = text.find("Introduction")
    abstract_text = text[abstract_start:abstract_end]

    return abstract_text
    

st.title("論文ヘルパー(英語のみ)")
uploaded_file = st.file_uploader("PDFファイルを選択してください", type="pdf")

if uploaded_file is not None:
    # ファイルを処理するために、一時的に保存されたPDFファイルのパスを使用
    # （例：pdf_text = read_pdf(temp_file_path)）
    temp_file_path = save_uploaded_file(uploaded_file)

    #PDFの読み込み
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()
    abstract_text = extract_text_from_pdf(temp_file_path)
   
    # 処理が終わったら、一時ファイルを削除
    os.unlink(temp_file_path)

    # ラジオボタンで「要約」や「質問」の選択肢を作成
    option = st.radio("実行したい処理を選択してください", ("Abstractの翻訳","要約", "質問"))

 # 実行ボタンを作成
    if option == "Abstractの翻訳":
        st.write("Abstractを抽出して翻訳を行います")
        if st.button("実行"):

            #LLMの定義
            llm_abstract = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)

            #abstractを翻訳するプロンプト
            template = """
                    あなたは優秀な翻訳家です。
                    以下の文章を英語から日本語に翻訳してください。

                    文章：{abstract}
                    """

            prompt_template = PromptTemplate(
                    input_variables=['abstract'], 
                    template=template
                    )

            abstract_chain = LLMChain(
                    llm=llm_abstract, 
                    prompt=prompt_template, 
                    output_key="translation" #output_keyを指定して次のチェインに入力する
                    )
            
            #まとめを作成するプロンプト
            template = """
                    あなたは優秀なアシスタントです。
                    以下の文章を一言でまとめてください。

                    文章：{translation}
                    """

            prompt_template = PromptTemplate(
                    input_variables=['translation'], 
                    template=template
                    )

            translation_chain = LLMChain(
                    llm=llm_abstract, 
                    prompt=prompt_template, 
                    output_key="single_word"
                    )
            
            #chainの作成
            overall_chain = SequentialChain(
                    chains=[abstract_chain, translation_chain],
                    input_variables=['abstract'],
                    output_variables=["translation", "single_word"], #複数の変数を返す
                    )

            #辞書型で定義する
            sentence = overall_chain({'abstract': abstract_text})

            #各文章を取得
            abstract = sentence['abstract']
            translation = sentence['translation']
            single_word = sentence['single_word']

            st.write(f'原文：{abstract}')
            st.write(f'翻訳：{translation}')
            st.write(f'まとめ：{single_word}')



    if option == "要約":
        st.write("論文全体の要約を行います")
        st.write("かなり時間かかります……")

        # ユーザーにパラメータを入力させる
        temperature = st.slider('出力の多様性', 0.0, 1.0, 0.0) 
        chunk_size = st.slider('分割する文字数', 1000, 4000, 4000) 
        chunk_overlap = st.slider('重複する文字数', 0, 50, 20) 

        if st.button("実行"):

            #テンプレートの準備
            prompt_template_summary = """
                            日本語で回答してください。
                            以下の内容を要約してください。
                            内容:{text}
                            回答：
                             """
            
            prompt_summary = PromptTemplate(
                template=prompt_template_summary, 
                input_variables=["text"]
                )
            
            #要約のチェインを作成
            llm_summary = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature)
            text_splitter_summary = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            texts_summary = text_splitter_summary.split_documents(documents)
            chain = load_summarize_chain(
                            llm=llm_summary, 
                            chain_type="map_reduce",
                            return_intermediate_steps=False,
                            map_prompt=prompt_summary, 
                            combine_prompt=prompt_summary
                            )
            
            #要約された文章を表示
            sammary = chain.run(texts_summary)
            st.write(sammary)



    elif option == "質問":

        #埋め込み型の作成
        text_splitter_qa = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts_qa = text_splitter_qa.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(texts_qa, embeddings)

        st.title("Q & A")
        with st.form("question_form"):
            user_input = st.text_input("質問したい内容を記載してください:")

            if st.form_submit_button("送信"):
                if user_input:  # user_inputが空でないかチェック
                    st.write(f"入力されたテキスト：{user_input}")

                    #レトリーバーの作成
                    llm_qa=ChatOpenAI(model_name="gpt-3.5-turbo")
                    retriever = vectordb.as_retriever()
                    qa = RetrievalQA.from_chain_type(llm=llm_qa, chain_type="stuff", retriever=retriever)

                    #テンプレートの用意
                    template_qa = """
                                あなたは優秀なアシスタントです。
                                論文に記載されている内容について回答してください。
                                日本語で回答してください。
                                質問：{question}
                                回答：
                                """

                    prompt_qa = PromptTemplate(
                                input_variables=["question"],
                                template=template_qa,
                                )
                    #プロンプトの作成
                    prompt_text = prompt_qa.format(question=user_input)
                    query = prompt_text

                    #回答の表示
                    anser = qa.run(query)
                    st.write(anser)

                else:
                    st.warning("テキストが入力されていません。")


            else:
                st.write("テキストボックスに入力して、送信ボタンを押してください。")
            
        
