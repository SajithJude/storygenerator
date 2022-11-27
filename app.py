import urllib
from random import randint
import torch
from transformers import pipeline, set_seed
from transformers.pipelines import TextGenerationPipeline
import streamlit as st
# import SessionState
# from SessionState import _SessionState, _get_session, _get_state
import logging


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# def load_bad_words() -> list:
#     res_list = []

#     try:
#         file = urllib.request.urlopen(
#             "https://raw.githubusercontent.com/coffee-and-fun/google-profanity-words/main/data/list.txt"
#         )
#         for line in file:
#             dline = line.decode("utf-8")
#             res_list.append(dline.split("\n")[0])
#     except:
#         logging.info("Failed to load bad words list.")

#     return res_list


# BAD_WORDS = load_bad_words()

STARTERS = {
    0: "Rick: Morty, quick! Get in the car!\nMorty: Oh no, I can't do it Rick! Please not this again.\nRick: You don't have a choice! The crystal demons are going to eat you if you don't get in!",
    1: "Elon: Oh, you think you're all that Rick? Fight me in a game of space squash!\nRick: Let's go, you wanna-be genius!\nElon: SpaceX fleet, line up!",
    2: "Morty: I love Jessica, I want us to get married on Octopulon 300 and have octopus babies.\nRick: Shut up, Morty! You're not going to Octopulon 300!",
    3: "Rick: Hey there, Jerry! What a nice day for taking these anti-gravity shoes for a spin!\nJerry: Wow, Rick! You would let me try out one of your crazy gadgets?\nRick: Of course, Jerry! That's how much I respect you.",
    4: "Rick: Come on, flip the pickle, Morty. You're not gonna regret it. The payoff is huge.",
    5: "Rick: I turned myself into a pickle, Morty! Boom! Big reveal - I'm a pickle. What do you think about that? I turned myself into a pickle!",
    6: "Rick: Come on, flip the pickle, Morty. You're not gonna regret it. The payoff is huge.\nMorty: What? Where are you?\nRick: Morty, just do it! [laughing] Just flip the pickle!",
}


@st.cache(allow_output_mutation=True, suppress_st_warning=True,max_entries=1)
def load_model() -> TextGenerationPipeline:
    return pipeline("text-generation", model="e-tony/gpt2-rnm")



def main():
    # state = st.session_state.
    st.set_page_config(page_title="Story Generator", page_icon="🛸")

    model = load_model()
    # set_seed(42)  # for reproducibility

    load_page(model)

    # state.sync()  # Mandatory to avoid rollbacks with widgets, must be called at the end of your app


def load_page(model: TextGenerationPipeline):
    st.write("---")

    st.title("Story Generator")
    # icon = "https://static.wikia.nocookie.net/rickandmorty/images/7/77/Butter_Robot.png/revision/latest?cb=20160910011723"
    st.image("Butter_Robot.jpg")

    slider = st.slider(
        "Set your story's length (longer scripts will take more time to generate):",
        50,
        1000,
        
    )

    input = st.text_area(
        "Start your story:",
        STARTERS[randint(0, 6)],
        height=100,
        max_chars=5000,
    )



    if len(input) + slider > 5000:
        st.warning("Your story cannot be longer than 5000 characters!")
        st.stop()

    button_generate = st.button("Generate Story (burps)")
    # if st.button("Reset Prompt (Random)"):
    #     state.clear()

    if button_generate:
        try:
            outputs = model(
                input,
                do_sample=True,
                max_length=len(input) + slider,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
            )
            output_text = outputs[0]["generated_text"]
            input = st.text_area(
                "Start your story:", output_text or "", height=50
            )
        except:
            pass

    st.markdown(
        '<h2 style="font-family:Courier;text-align:center;">And heres what happened later</h2>',
        unsafe_allow_html=True,
    )

    for i, line in enumerate(input.split("\n")):
        if ":" in line:
            speaker, speech = line.split(":")

            st.markdown(
                f'<p style="font-family:Courier;text-align:center;"><b>{speaker}:</b><br>{speech}</br></p>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<p style="font-family:Courier;text-align:center;">{line}</p>',
                unsafe_allow_html=True,
            )
    
    st.markdown("---")
    # st.markdown(
    #     "_You can read about how to create your own story generator application [here](https://towardsdatascience.com/rick-and-morty-story-generation-with-gpt2-using-transformers-and-streamlit-in-57-lines-of-code-8f81a8f92692). The code for this project is on [Github](https://github.com/e-tony/Story_Generator)._"
    # )


if __name__ == "__main__":
    main()