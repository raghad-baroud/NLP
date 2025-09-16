import re
import emoji
import pyarabic.araby as araby
import nltk
from nltk.tokenize import word_tokenize
from tashaphyne.stemming import ArabicLightStemmer
import plotly.express as px
import plotly.io as pio
import matplotlib as mpl
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import arabic_reshaper
from bidi.algorithm import get_display
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from PIL import Image
import streamlit as st
import pandas as pd
import numpy as np 
import pickle
from pathlib import Path
import torch
from .models import get_dataloader, get_model, infer_sentiments
import time

#emojis encoding

arabic_emoji = pd.read_csv(f"{Path(__file__).parent}\\arabic_emojis.csv")
UNICODE_EMOJI = dict(map(lambda i,j : (i,j) , list(arabic_emoji['emoji']),list(arabic_emoji['text'])))

#stopwords
stop_words = set(stopwords.words('arabic')).union({"،","","ورحمه","وبركاته","عليكم","السلام","آض","آمينَ","آه","آهاً","آي","أ","أب","أجل","أجمع","أخ","أخذ","أصبح","أضحى","أقبل",
"أقل","أكثر","ألا","أم","أما","أمامك","أمامكَ","أمسى","أمّا","أن","أنا","أنت","أنتم","أنتما","أنتن","أنتِ","أنشأ","أنّى","أو","أوشك","أولئك",
"أولئكم","أولاء","أولالك","أوّهْ","أي","أيا","أين","أينما","أيّ","أَنَّ","أََيُّ","أُفٍّ","إذ","إذا","إذاً","إذما","إذن","إلى","إليكم","إليكما","إليكنّ",
"إليكَ","إلَيْكَ","إلّا","إمّا","إن","إنّما","إي","إياك","إياكم","إياكما","إياكن","إيانا","إياه","إياها","إياهم","إياهما","إياهن","إياي","إيهٍ","إِنَّ","ا","ابتدأ",
"اثر","اجل","احد","اخرى","اخلولق","اذا","اربعة","ارتدّ","استحال","اطار","اعادة","اعلنت","اف","اكثر","اكد","الألاء","الألى","الا","الاخيرة","الان",
"الاول","الاولى","التى","التي","الثاني","الثانية","الذاتي","الذى","الذي","الذين","السابق","الف","اللائي","اللاتي","اللتان","اللتيا","اللتين",
"اللذان","اللذين","اللواتي","الماضي","المقبل","الوقت","الى","اليوم","اما","امام","امس","ان","انبرى","انقلب","انه","انها","او","اول","اي",
"ايار","ايام","ايضا","ب","بات","باسم","بان","بخٍ","برس","بسبب","بسّ","بشكل","بضع","بطآن","بعد","بعض","بك","بكم","بكما","بكن","بل","بلى","بما",
"بماذا","بمن","بن","بنا","به","بها","بي","بيد","بين","بَسْ","بَلْهَ","بِئْسَ","تانِ","تانِك","تبدّل","تجاه","تحوّل","تلقاء","تلك","تلكم","تلكما","تم",
"تينك","تَيْنِ","تِه","تِي","ثلاثة","ثم","ثمّ","ثمّة","ثُمَّ","جعل","جلل","جميع","جير","حار","حاشا","حاليا","حاي","حتى","حرى","حسب","حم","حوالى","حول",
"حيث","حيثما","حين","حيَّ","حَبَّذَا","حَتَّى","حَذارِ","خلا","خلال","دون","دونك","ذا","ذات","ذاك","ذانك","ذانِ","ذلك","ذلكم","ذلكما","ذلكن","ذو","ذوا",
"ذواتا","ذواتي","ذيت","ذينك","ذَيْنِ","ذِه","ذِي","راح","رجع","رويدك","ريث","رُبَّ","زيارة","سبحان","سرعان","سنة","سنوات","سوف","سوى","سَاءَ","سَاءَمَا",
"شبه","شخصا","شرع","شَتَّانَ","صار","صباح","صفر","صهٍ","صهْ","ضمن","طاق","طالما","طفق","طَق","ظلّ","عاد","عام","عاما","عامة","عدا","عدة","عدد","عدم",
"عسى","عشر","عشرة","علق","على","عليك","عليه","عليها","علًّ","عن","عند","عندما","عوض","عين","عَدَسْ","عَمَّا","غدا","غير","ـ","ف","فان","فلان","فو",
"فى","في","فيم","فيما","فيه","فيها","قال","قام","قبل","قد","قطّ","قلما","قوة","كأنّما","كأين","كأيّ","كأيّن","كاد","كان","كانت","كذا","كذلك","كرب",
"كل","كلا","كلاهما","كلتا","كلم","كليكما","كليهما","كلّما","كلَّا","كم","كما","كي","كيت","كيف","كيفما","كَأَنَّ","كِخ","لئن","لا","لات","لاسيما","لدن","لدى",
"لعمر","لقاء","لك","لكم","لكما","لكن","لكنَّما","لكي","لكيلا","للامم","لم","لما","لمّا","لن","لنا","له","لها","لو","لوكالة","لولا","لوما","لي","لَسْتَ",
"لَسْتُ","لَسْتُم","لَسْتُمَا","لَسْتُنَّ","لَسْتِ","لَسْنَ","لَعَلَّ","لَكِنَّ","لَيْتَ","لَيْسَ","لَيْسَا","لَيْسَتَا","لَيْسَتْ","لَيْسُوا","لَِسْنَا","ما","ماانفك","مابرح","مادام","ماذا",
"مازال","مافتئ","مايو","متى","مثل","مذ","مساء","معاذ","مقابل","مكانكم","مكانكما","مكانكنّ","مكانَك","مليار","مليون","مما","ممن","من","منذ",
"منها","مه","مهما","مَنْ","مِن","نحن","نحو","نعم","نفس","نفسه","نهاية","نَخْ","نِعِمّا","نِعْمَ","ها","هاؤم","هاكَ","هاهنا","هبّ","هذا","هذه","هكذا",
"هل","هلمَّ","هلّا","هم","هما","هن","هنا","هناك","هنالك","هو","هي","هيا","هيت","هيّا","هَؤلاء","هَاتانِ","هَاتَيْنِ","هَاتِه","هَاتِي","هَجْ","هَذا","هَذانِ","هَذَيْنِ",
"هَذِه","هَذِي","هَيْهَاتَ","و","و6","وا","واحد","واضاف","واضافت","واكد","وان","واهاً","واوضح","وراءَك","وفي","وقال","وقالت","وقد","وقف","وكان","وكانت",
"ولا","ولم","ومن","مَن","وهو","وهي","ويكأنّ","وَيْ","وُشْكَانََ","يكون","يمكن","يوم","ّأيّان"})

def _clean_data(text):
  text=str(text)
  #remove repeated words of  ا و ي ح خ ر
  text=re.sub(r'([اويخحر])\1+', r'\1', text)
  #remove urls
  text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', ' ', text)
  #remove punctuations
  text = re.sub('[_]', ' ',  text)
  text = re.sub(r'[!()-{};:\,<>./?@#$%^&*_~\n\t""،]',' ',text)
  #remove non-arabic characters & digits
  #text = re.sub(r'[A-Za-z0-9٠-٩]+', '', text)
  text =re.sub(r'[^0-9\u0600-\u06ff\u0750-\u077f\ufb50-\ufbc1\ufbd3-\ufd3f\ufd50-\ufd8f\ufd50-\ufd8f\ufe70-\ufefc\uFDF0-\uFDFD.0-9٠-٩]+', ' ',text)
  #remove @usernames
  text = re.sub('@[^\s]+', ' ', text)
  #remove laughs of any long and normalized it to ههه
  text= re.sub('[ه]{3,}', 'ههه', text)
  #remove diacritics
  text = araby.strip_diacritics(text)
  #remove tashkeel
  text = araby.strip_tashkeel(text)
  #remove tatweel
  text = araby.strip_tatweel(text)
  #remove stop
  tokens=word_tokenize(text)
  text=" ".join([w.strip() for w in tokens if not w in stop_words and len(w) >= 2])

  #convert emoji to text
  for emot in UNICODE_EMOJI:
      text = ''.join("_".join(UNICODE_EMOJI[emot].replace(",", "").replace(":", "").split()) + " " if emot in UNICODE_EMOJI else emot for emot in text)
  #remove other emojis
  text = emoji.replace_emoji(text, replace='')
  
  return text


def _stem(text):
  str(text)
  stemmer=ArabicLightStemmer()
  tokens=word_tokenize(text)
  tokens
  stemmized=[]
  for t in tokens:
    stem=stemmer.light_stem(t)
    stemmized.append(stem)

  return ' '.join(stemmized)

def get_device():
   device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   return device

def get_classification(dataset):
  data=dataset.values
  device=get_device()
  #load dataloader
  dataloader=get_dataloader(data)
  #load the models
  model=get_model(device,type='lstm')
  #make predictions
  start=time.time()
  predictions=infer_sentiments(model, dataloader,device,type='lstm')
  inference_time=time.time() - start
  print(f'It took the attn-LSTM: {inference_time} to classfy data.')
  mapped_predictions={0:'negative',1:'neutral',2:'positive'}
  y_pred=[mapped_predictions[i] for i in predictions]
  data=map(_clean_data,data) #pass cleaned data
  return pd.DataFrame({'Tweet': data, 'Class': y_pred})

def pieplot_sentiment(dataset):
  sentiment_count = dataset["Class"].value_counts()

  # plot the sentiment distribution in a pie chart
  fig = px.pie(
          values=sentiment_count.values,
          names=sentiment_count.index,
          hole=0.3,
          title="<b>Sentiment Distribution</b>",
          color=sentiment_count.index,
          # set the color of positive to blue and negative to orange
          color_discrete_map={"positive": "#1F77B4", "negative": "#FF7F0E"},
      )
  fig.update_traces(
          textposition="inside",
          texttemplate="%{label}<br>%{value} (%{percent})",
          hovertemplate="<b>%{label}</b><br>Percentage=%{percent}<br>Count=%{value}",
      )
  fig.update_layout(showlegend=False)
  return fig

def plot_wordcloud(dataset, colormap="Greens"):
    #stopwords
    stopwords=stop_words.union({'اكسبو','دبي','إكسبو'})

    # load the mask image and font type
    mask = np.array(Image.open(f"{Path(__file__).parent}\\twitter_mask.png"))
    font = f"{Path(__file__).parent}\\NotoNaskhArabic-Regular.ttf"

    # generate custom colormap
    cmap = mpl.cm.get_cmap(colormap)(np.linspace(0, 1, 20))
    cmap = mpl.colors.ListedColormap(cmap[10:15])

    # combine all the preprocessed tweets into a single string
    text = " ".join(dataset['Tweet'])

    text = arabic_reshaper.reshape(text)
    text = get_display(text)

    # create the WordCloud instance
    wc = WordCloud(
        background_color="white",
        font_path=font,
        stopwords =stopwords,
        max_words=90,
        colormap=cmap,
        mask=mask,
        random_state=42,
        collocations=False,
        min_word_length=2,
        max_font_size=200,
    )

    # generate and plot the wordcloud
    wc.generate(text)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Wordcloud", fontdict={"fontsize": 16}, fontweight="heavy", pad=20, y=1.0)
    return fig

def _get_top_n_gram(dataset, ngram_range, n=10):
    # stopwords
    stopwords = stop_words.union({"اكسبو", "دبي",'إكسبو'})
    # load the corpus and vectorizer
    corpus = dataset['Tweet']
    
    vectorizer = CountVectorizer(
            analyzer="word", ngram_range=ngram_range, stop_words=list(stopwords)
        )

    # use the vectorizer to count the n-grams frequencies
    X = vectorizer.fit_transform(corpus.astype(str).values)
    words = vectorizer.get_feature_names_out()
    words_count = np.ravel(X.sum(axis=0))

    # store the results in a dataframe
    df = pd.DataFrame(zip(words, words_count))
    df.columns = ["words", "counts"]
    df = df.sort_values(by="counts", ascending=False).head(n)
    df["words"] = df["words"].str.title()
    return df

def _plot_n_gram(n_gram_df, title, color="#54A24B"):
    # plot the top n-grams frequencies in a bar chart
    fig = px.bar(
        x=n_gram_df.counts,
        y=n_gram_df.words,
        title="<b>{}</b>".format(title),
        text_auto=True,
    )
    fig.update_layout(plot_bgcolor="white")
    fig.update_xaxes(title=None)
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_traces(hovertemplate="<b>%{y}</b><br>Count=%{x}", marker_color=color)
    return fig

def get_top_occuring_words_graph(dataset, ngram_range, title, n=10, color="#54A24B"):
    n_gram_df=_get_top_n_gram(dataset, ngram_range,n)
    fig=_plot_n_gram(n_gram_df, title, color="#54A24B")
    return fig
   
