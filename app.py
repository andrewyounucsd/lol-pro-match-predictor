import streamlit as st
import pickle
import numpy as np

# load artifacts
with open('model_artifacts.pkl', 'rb') as f:
    artifacts = pickle.load(f)

model = artifacts['model']
node2vec = artifacts['node2vec']
champion_list = artifacts['champion_list']

st.title('LoL Pro Match Predictor')
st.markdown('Select 5 champions for each team to predict the winner.')

col1, col2 = st.columns(2)

with col1:
    st.subheader('Blue Side')
    blue_picks = []
    for i in range(1, 6):
        pick = st.selectbox(f'Pick {i}', [''] + champion_list, key=f'blue_{i}')
        blue_picks.append(pick)

with col2:
    st.subheader('Red Side')
    red_picks = []
    for i in range(1, 6):
        pick = st.selectbox(f'Pick {i}', [''] + champion_list, key=f'red_{i}')
        red_picks.append(pick)

def team_embedding(picks, model):
    return np.sum([model.wv[p] for p in picks], axis=0)

if st.button('Predict'):
    if '' in blue_picks or '' in red_picks:
        st.error('Please select all 10 champions.')
    elif len(set(blue_picks + red_picks)) != 10:
        st.error('Each champion can only be picked once.')
    else:
        blue_emb = team_embedding(blue_picks, node2vec)
        red_emb = team_embedding(red_picks, node2vec)
        X = np.hstack([blue_emb, red_emb]).reshape(1, -1)
        
        prob = model.predict_proba(X)[0]
        blue_prob = prob[1]
        red_prob = prob[0]

        st.markdown('---')
        st.subheader('Prediction')
        
        col3, col4 = st.columns(2)
        with col3:
            st.metric('Blue Side Win Probability', f'{blue_prob:.1%}')
        with col4:
            st.metric('Red Side Win Probability', f'{red_prob:.1%}')

        if blue_prob > red_prob:
            st.success('Blue Side favored to win!')
        else:
            st.error('Red Side favored to win!')

        st.markdown('---')
        st.caption('Model: Gradient Boosting + Node2Vec Champion Embeddings trained on 2087 major region pro matches (2025)')
