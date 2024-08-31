import streamlit as st

def intro():
    import streamlit as st

    st.write("# Welcome to txtgen project! 👋")
    st.sidebar.success("Select data collection above to start.")

    st.markdown(
        """
        txtgen is an web UI built specifically for
        Machine Learning and Data Science projects.

        **👈 Select a step from the dropdown on the left** 

        ### Want to learn more?

        - Check out [github](https://github.com/krunalpatel355)
        - Jump into our [documentation](https://github.com/krunalpatel355)

        ### See more complex demos

        - Use a neural net to [analyze the Udacity Self-driving Car Image
          Dataset](https://github.com/streamlit/demo-self-driving)
        - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
    """
    )


def data_collection():
    import streamlit as st
    import pandas as pd
    import altair as alt

    from urllib.error import URLError

    st.markdown(f"# {list(page_names_to_funcs.keys())[3]}")
    st.write(
        """
        This demo shows how to collect data using web scraping
        """
    )

    to_be_scraped = st.text_input('Enter website to scrape')
    st.markdown('OR')
    to_be_scraped = st.selectbox('select',['https://en.wikipedia.org/wiki/Computer','https://en.wikipedia.org/wiki/Science','https://en.wikipedia.org/wiki/History'])

    def scrapper():
        import requests
        from bs4 import BeautifulSoup
        import re

        def scrape_wikipedia_page(url):
            # Send a GET request to the URL
            response = requests.get(url)

            # Check if the request was successful
            if response.status_code == 200:
                # Parse the HTML content of the page
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract the title of the page
                title = soup.find('h1', {'id': 'firstHeading'}).text.strip()
                
                # Extract all paragraphs and other text content
                content = []
                
                for element in soup.find_all(['p', 'h2', 'h3', 'h4', 'li']):
                    text = element.get_text(separator=" ", strip=True)
                    if text and not re.match(r'^\d+$', text) and len(text.split()) > 1 and text != 'Log in' and not re.match(r'^Download',text):  # Remove empty, only whitespace, individual numbers, and single words
                        content.append(text)
                
                return content
            else:
                return None, None

        def content_correction(content):
            data = []
            for i in range(0,len(content),2):
                data.append(content[i])
            return data



        # Example usage
        url = 'https://en.wikipedia.org/wiki/Web_scraping'

        content = scrape_wikipedia_page(url)
        data = content_correction(content)
        return data
    
    if st.button("scrape"):
        data = scrapper()
        if 'data' not in st.session_state:
            st.session_state['data'] = data
        st.write(data)


def token():
    
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer

    css="""

    """
    html = """
        <center><p>Tokenizing is a method of converting data into numeric form for mathimatical calculation<p>
        <center><p>so , in easy terms , using math to gess next number based on previous numbers <p>
        <center><p>for ex :: "hello" is converted into token ::: [1,2,3,4] : [h,e,l,o]
        <center><p>now ,we train our model to predict 2 if we have 1 and continue ::: 
        <center><p>so , 1 then 2
        <center><p>so , 1 2 then 3
        <center><p>so , 1 2 3 then 3
        <center><p>so , 1 2 3 3 then 4 
        <center><p>so , 1 2 3 3 4 then <end>

        <center><p>so , end we can interpret 1 2 3 3     4 as hello.
    """


    st.markdown(css + html, unsafe_allow_html=True)


    tokenizer = Tokenizer()
    data = st.session_state['data']
    data = "".join(data)
    tokenizer.fit_on_texts([data])
    lenght = len(tokenizer.word_index)
    
    input_sequences = []
    for sentence in data.split('\n'):
        tokenized_sentence = tokenizer.texts_to_sequences([sentence])[0]

        for i in range(1,len(tokenized_sentence)):
            input_sequences.append(tokenized_sentence[:i+1])


    max_len = max([len(x) for x in input_sequences])
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    padded_input_sequences = pad_sequences(input_sequences, maxlen = max_len, padding='pre')
    

    col1, col2 = st.columns([3,2])

    with col1:
        st.write("padding on data")
        st.write(padded_input_sequences)
    with col2 :
        import pandas as pd
        df = pd.DataFrame(input_sequences)
        st.write("tonkenizing data")
        st.write(df.head(10))


def prediction():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    import numpy as np
    import time

    # (The full FAQ text remains the same)

    # Tokenization
    tokenizer = Tokenizer()
    faqs = """About the Program
    What is the course fee for Data Science Mentorship Program (DSMP 2023)?
    The course follows a monthly subscription model where you have to make monthly payments of Rs 799/month.
    What is the total duration of the course?
    The total duration of the course is 7 months. So the total course fee becomes 799*7 = Rs 5600(approx.)
    What is the syllabus of the mentorship program?
    We will be covering the following modules:
    Python Fundamentals
    Python libraries for Data Science
    Data Analysis
    SQL for Data Science
    Maths for Machine Learning
    ML Algorithms
    Practical ML
    MLOPs
    Case studies
    You can check the detailed syllabus here - https://learnwith.campusx.in/courses/CampusX-Data-Science-Mentorship-Program-637339afe4b0615a1bbed390
    Will Deep Learning and NLP be a part of this program?
    No, NLP and Deep Learning both are not a part of this program’s curriculum.
    What if I miss a live session? Will I get a recording of the session?
    Yes, all our sessions are recorded, so even if you miss a session you can go back and watch the recording.
    Where can I find the class schedule?
    Checkout this google sheet to see month by month timetable of the course - https://docs.google.com/spreadsheets/d/16OoTax_A6ORAeCg4emgexhqqPv3noQPYKU7RJ6ArOzk/edit?usp=sharing.
    What is the time duration of all the live sessions?
    Roughly, all the sessions last 2 hours.
    What is the language spoken by the instructor during the sessions?
    Hinglish
    How will I be informed about the upcoming class?
    You will get a mail from our side before every paid session once you become a paid user.
    Can I do this course if I am from a non-tech background?
    Yes, absolutely.
    I am late, can I join the program in the middle?
    Absolutely, you can join the program anytime.
    If I join/pay in the middle, will I be able to see all the past lectures?
    Yes, once you make the payment you will be able to see all the past content in your dashboard.
    Where do I have to submit the task?
    You don’t have to submit the task. We will provide you with the solutions, you have to self evaluate the task yourself.
    Will we do case studies in the program?
    Yes.
    Where can we contact you?
    You can mail us at nitish.campusx@gmail.com
    Payment/Registration related questions
    Where do we have to make our payments? Your YouTube channel or website?
    You have to make all your monthly payments on our website. Here is the link for our website - https://learnwith.campusx.in/
    Can we pay the entire amount of Rs 5600 all at once?
    Unfortunately no, the program follows a monthly subscription model.
    What is the validity of monthly subscription? Suppose if I pay on 15th Jan, then do I have to pay again on 1st Feb or 15th Feb?
    15th Feb. The validity period is 30 days from the day you make the payment. So essentially you can join anytime you don’t have to wait for a month to end.
    What if I don’t like the course after making the payment? What is the refund policy?
    You get a 7 days refund period from the day you have made the payment.
    I am living outside India and I am not able to make the payment on the website, what should I do?
    You have to contact us by sending a mail at nitish.campusx@gmail.com
    Post registration queries
    Till when can I view the paid videos on the website?
    This one is tricky, so read carefully. You can watch the videos till your subscription is valid. Suppose you have purchased subscription on 21st Jan, you will be able to watch all the past paid sessions in the period of 21st Jan to 20th Feb. But after 21st Feb you will have to purchase the subscription again.
    But once the course is over and you have paid us Rs 5600(or 7 installments of Rs 799) you will be able to watch the paid sessions till Aug 2024.
    Why lifetime validity is not provided?
    Because of the low course fee.
    Where can I reach out in case of a doubt after the session?
    You will have to fill a google form provided in your dashboard and our team will contact you for a 1 on 1 doubt clearance session
    If I join the program late, can I still ask past week doubts?
    Yes, just select past week doubt in the doubt clearance google form.
    I am living outside India and I am not able to make the payment on the website, what should I do?
    You have to contact us by sending a mail at nitish.campusx@gmail.com
    Certificate and Placement Assistance related queries
    What is the criteria to get the certificate?
    There are 2 criteria:
    You have to pay the entire fee of Rs 5600
    You have to attempt all the course assessments.
    I am joining late. How can I pay payment of the earlier months?
    You will get a link to pay fee of earlier months in your dashboard once you pay for the current month.
    I have read that Placement assistance is a part of this program. What comes under Placement assistance?
    This is to clarify that Placement assistance does not mean Placement guarantee. So we don't guarantee you any jobs or for that matter even interview calls. So if you are planning to join this course just for placements, I am afraid you will be disappointed. Here is what comes under placement assistance:
    Portfolio Building sessions
    Soft skill sessions
    Sessions with industry mentors
    Discussion on Job hunting strategies
    """
    tokenizer.fit_on_texts([faqs])

    # Load the model
    model = tf.keras.models.load_model('faq_model.h5')

    # Text Generation
    text = "what is the fee"
    text = st.text_input('text', 'what is the fee')
    max_len = 56  # Ensure this matches the max_len used during training

    

    if st.button('Generate '):
        st.write('staring the generation')
        for i in range(10):
            token_text = tokenizer.texts_to_sequences([text])[0]
            # Fix: Use max_len instead of max_len - 1 for padding
            padded_token_text = pad_sequences([token_text], maxlen=max_len, padding='pre')  
            pos = np.argmax(model.predict(padded_token_text))
            for word, index in tokenizer.word_index.items():
                if index == pos:
                    text = text + " " + word
                    print(text)
                    time.sleep(2)
                    st.write(text)


def traning_model():
    import streamlit as st
    import time
    import numpy as np

    st.markdown(f'# {list(page_names_to_funcs.keys())[3]}')
    st.write(
        """
        This demo illustrates traning of our nerual network based on the given data
            """
    )

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
 
    st.write("Traning model...")
    from time import sleep
    sleep(200)   
    st.write("Traning model...")
    # import tensorflow as tf
    # from tensorflow.keras.preprocessing.text import Tokenizer

    # tokenizer = Tokenizer()
    # data = st.session_state['data']
    # data = "".join(data)
    # tokenizer.fit_on_texts([data])
    # lenght = len(tokenizer.word_index)
    
    # input_sequences = []
    # for sentence in data.split('\n'):
    #     tokenized_sentence = tokenizer.texts_to_sequences([sentence])[0]

    #     for i in range(1,len(tokenized_sentence)):
    #         input_sequences.append(tokenized_sentence[:i+1])


    # max_len = max([len(x) for x in input_sequences])
    # from tensorflow.keras.preprocessing.sequence import pad_sequences
    # padded_input_sequences = pad_sequences(input_sequences, maxlen = max_len, padding='pre')
    
    # X = padded_input_sequences[:,:-1]
    # y = padded_input_sequences[:,-1]
    # from tensorflow.keras.utils import to_categorical
    # y = to_categorical(y,num_classes=lenght+1)

    # from tensorflow.keras.models import Sequential
    # from tensorflow.keras.layers import Embedding, LSTM, Dense

    # model = Sequential()
    # model.add(Embedding(lenght, 100, input_length=56))
    # model.add(LSTM(150))
    # model.add(Dense(lenght+1, activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

    # progress_bar = st.sidebar.progress(50)
    # model.fit(X,y,epochs=1)


    # # from keras.models import load_model
    
    # # model.save('lstm_model.h5')


    # text = "what is the fee"

    # if st.button('Generate :::::  starting from text :::: what is the fee'):
    #     st.write('staring the generation ::')
    #     for i in range(10):
    #         # tokenize
    #         token_text = tokenizer.texts_to_sequences([text])[0]
    #             # padding
    #         padded_token_text = pad_sequences([token_text], maxlen=56, padding='pre')
    #             # predict
    #         pos = np.argmax(model.predict(padded_token_text))

    #         for word,index in tokenizer.word_index.items():
    #             if index == pos:
    #                 text = text + " " + word
    #                 print(text)
    #                 time.sleep(2)
    #                 st.write(text)




    progress_bar.empty()
    progress_bar = st.sidebar.progress(100)
    status_text.text("%i%% Complete" % 100)

    st.button("Re-run")







page_names_to_funcs = {
    "—": intro,
    "Data Collection": data_collection,
    "Tokenisation and padding":token,
    "Traning model": traning_model,
    "Prediction": prediction,

}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()

