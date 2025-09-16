import streamlit as st
from assets import utils

# function to color the sentiment column
def sentiment_color(sentiment):
    if sentiment == "positive":
        return "background-color: #1F77B4; color: white"
    else:
        return "background-color: #FF7F0E"
    
if "df" in st.session_state:
        st.session_state.df=utils.get_classification(st.session_state.df)
        # increase the font size of text inside the tabs
        adjust_tab_font = """
            <style>
            button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
                font-size: 20px;
            }
            </style>
            """
        st.write(adjust_tab_font, unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(["All", "Positive 😊", "Negative ☹️"])
        with tab1:
            # make dashboard for all tweets
            tweet_df = st.session_state.df.copy() #take a copy
            bar_color, wc_color="#54A24B", "Greens"
            col1, col2 = st.columns([50, 50])
            with col1:
                # plot the sentiment distribution
                sentiment_plot = utils.pieplot_sentiment(tweet_df)
                sentiment_plot.update_layout(height=350, title_x=0.5)
                st.plotly_chart(sentiment_plot, theme=None, use_container_width=True)

            with col2:
                # plot the top 10 occuring words
                unigram_plot=utils.get_top_occuring_words_graph(tweet_df,
                                                            ngram_range=(1,1),
                                                            title="Top 10 Occuring Words",
                                                            n=10, 
                                                            color=bar_color) 
                unigram_plot.update_layout(height=350)
                st.plotly_chart(unigram_plot, theme=None, use_container_width=True)





            # show the dataframe containing the tweets and their sentiment
            st.dataframe(
                    tweet_df[["Class", "Tweet"]].style.applymap(
                        sentiment_color, subset=["Class"]
                        ),
                    height=350
                    )

        with tab2:
                # make dashboard for tweets with positive sentiment
                tweet_df = st.session_state.df.query("Class == 'positive'")
                bar_color, wc_color="#1F77B4","Blues"
                col1, col2= st.columns([50, 50])
                with col1:
                    # plot the wordcloud
                    wordcloud = utils.plot_wordcloud(tweet_df, colormap=wc_color)
                    st.pyplot(wordcloud)
                with col2:
                    
                    # plot the top 10 occuring words
                    unigram_plot=utils.get_top_occuring_words_graph(tweet_df,
                                                            ngram_range=(1,1),
                                                            title="Top 10 Occuring Words",
                                                            n=10, 
                                                            color=bar_color) 
                    unigram_plot.update_layout(height=350)
                    st.plotly_chart(unigram_plot, theme=None, use_container_width=True)

                # show the dataframe containing the tweets and their sentiment
                st.dataframe(
                        tweet_df[["Class", "Tweet"]].style.applymap(
                            sentiment_color, subset=["Class"]
                        ),
                        height=350
                    )

        with tab3:
                # make dashboard for tweets with negative sentiment
                tweet_df = st.session_state.df.query("Class == 'negative'")
                bar_color,wc_color="#FF7F0E","Oranges"
                col1, col2= st.columns([50, 50])
                with col1:
                    # plot the wordcloud
                    wordcloud = utils.plot_wordcloud(tweet_df, colormap=wc_color)
                    st.pyplot(wordcloud)
                with col2:
                    
                    # plot the top 10 occuring words
                    unigram_plot=utils.get_top_occuring_words_graph(tweet_df,
                                                            ngram_range=(1,1),
                                                            title="Top 10 Occuring Words",
                                                            n=10, 
                                                            color=bar_color) 
                    unigram_plot.update_layout(height=350)
                    st.plotly_chart(unigram_plot, theme=None, use_container_width=True)

                # show the dataframe containing the tweets and their sentiment
                st.dataframe(
                        tweet_df[["Class", "Tweet"]].style.applymap(
                            sentiment_color, subset=["Class"]
                        ),
                        height=350
                    )
else: 
    st.error('Please upload the files to view the Analysis Report.')
