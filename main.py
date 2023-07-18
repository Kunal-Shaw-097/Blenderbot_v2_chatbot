import streamlit as st
from streamlit_chat import message
import torch
from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer

if "Question" not in st.session_state:
    model_id = "facebook/blenderbot-400M-distill"
    st.session_state.Question = []
    st.session_state.Response = []
    st.session_state.History = []
    st.session_state.Tokenizer = BlenderbotTokenizer.from_pretrained(model_id, truncation_side='left')
    if torch.cuda.is_available():
        st.session_state.model = BlenderbotForConditionalGeneration.from_pretrained(model_id).to('cuda')
    else:
        st.session_state.model = BlenderbotForConditionalGeneration.from_pretrained(model_id)

def predict(prompt):
    if len(st.session_state.History) != 0 :
        past_conv = "".join(st.session_state.History)
        prompt = past_conv + "  " + prompt
    encoded_input = st.session_state.Tokenizer([prompt], return_tensors='pt', max_length = 128)
    if torch.cuda.is_available():
        encoded_input = encoded_input.to('cuda')
    output = st.session_state.model.generate(**encoded_input, do_sample = True)
    answer = st.session_state.Tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    return answer

def infer():
    if st.session_state.input and st.session_state.input != " ":
        Response = predict(st.session_state.input)
        st.session_state["Question"].append(st.session_state.input)
        st.session_state["Response"].append(Response)
        if len(st.session_state["History"]) == 0:
            st.session_state["History"].append(st.session_state.input + '   ' + Response + '  ')
        else :
            st.session_state["History"].append('  ' + st.session_state.input + '   ' + Response + '  ')
        st.session_state.input = " "

st.title("Chatbot")
st.sidebar.radio('R:',[1,2])
chatbox = st.container()

prompt = st.text_input("Type Here :", key = 'input', on_change = infer)

if prompt:
    with chatbox:    
        for key,(i, j) in enumerate(zip(st.session_state["Question"], st.session_state["Response"])):
                message(i, is_user= True, key = key)
                message(j, key = 10001 + key)







