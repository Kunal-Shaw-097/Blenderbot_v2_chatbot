import streamlit as st
from streamlit_chat import message
from models.utils import Load_Model, Generate, model_info
import torch
from PIL import Image

@st.cache_resource
def load_model(model_name) : 
    return Load_Model(model_name)

def init():
    torch.cuda.empty_cache()
    if "Question" not in st.session_state:
        st.session_state.Question = []
        st.session_state.Response = []
        st.session_state.History = []
        st.session_state.model_name = 'Facebook_Blenderbot'
        st.session_state.Generator = Generate()
        
def re_init():
    st.session_state.Question = []
    st.session_state.Response = []
    st.session_state.History = []
    st.session_state.input = " "
    del st.session_state.Generator
    st.session_state.Generator = Generate()

def infer():
    if st.session_state.input and st.session_state.input != " ":
        Response = st.session_state.Generator.generate(model, st.session_state.input)
        st.session_state["Question"].append(st.session_state.input)
        st.session_state["Response"].append(Response)
        st.session_state.input = " "

def main():
    col1, col2 = st.columns([4,1])
    with col1 :
        st.title("🤖 Chatbot")
    with col2 :
        if  st.session_state.model_name == 'Facebook_Blenderbot' :
            img = Image.open("static/blenderbot.png")
        else :
            img = Image.open("static/dialogpt.png")
        img = img.resize((175, 100))
        st.image(img)
    with st.sidebar :
        model_name = st.radio('Select a Model :',['Facebook_Blenderbot','Microsoft_Dialogpt'], 
                              index = 0, key = 'model_name', on_change = re_init)
        
        with st.container() :
            st.markdown("<style>.custom_div{background-color: black;  border: 2px ; border-radius : 4px; text;  padding: 10px; margin : 0 0 20px 0}</style> \
                        <div class = custom_div>" + "About the Model :" +  model_info(st.session_state.model_name) +"</div>",
                         unsafe_allow_html= True)
            
        st.error("Note : Due to the small context length, the models are not able to have any context of past chat after 3-4 turns.")
        st.info("[Github Link](https://github.com/Kunal-Shaw-097/streamlit_chatbot_app)", icon = '👾')

    chatbox = st.container()

    with chatbox:    
        for key,(i, j) in enumerate(zip(st.session_state["Question"], st.session_state["Response"])):
                message(i, is_user= True, key = key)
                message(j, key = 10001 + key)

    prompt = st.text_input("Type Here :", key = 'input', on_change = infer)

if __name__ ==  "__main__":
    
    init()
    model = load_model(st.session_state.model_name)
    main()








