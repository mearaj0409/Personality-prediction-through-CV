import os
import pandas as pd
import numpy as np
from tkinter import *
from tkinter import filedialog
import tkinter.font as font
from pyresparser import ResumeParser
from sklearn import linear_model 
from difflib import SequenceMatcher
import docx2txt
import PyPDF2
import re
import tensorflow as tf
from transformers import BertTokenizer,TFBertModel
import tensorflow_addons as tfa
import webbrowser

class train_model:
    
    def train(self):
        data =pd.read_csv('training_dataset.csv')
        array = data.values

        for i in range(len(array)):
            if array[i][0]=="Male":
                array[i][0]=1
            else:
                array[i][0]=0


        df=pd.DataFrame(array)

        maindf =df[[0,1,2,3,4,5,6]]
        mainarray=maindf.values

        temp=df[7]
        train_y =temp.values
        
        self.mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg',max_iter =1000)
        self.mul_lr.fit(mainarray, train_y)
        
    def test(self, test_data):
        try:
            test_predict=list()
            for i in test_data:
                test_predict.append(int(i))
            y_pred = self.mul_lr.predict([test_predict])
            return y_pred
        except:
            print("All Factors For Finding Personality Not Entered!")


def check_type(data):
    if type(data)==str or type(data)==str:
        return str(data).title()
    if type(data)==list or type(data)==tuple:
        str_list=""
        for i,item in enumerate(data):
            str_list+=item+", "
        return str_list
    else:   return str(data)

def getMBTIscore(cv_path):

    def extract_text_from_pdf(file_path):
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfFileReader(file)
            text = ""
            for page in range(pdf_reader.getNumPages()):
                page_obj = pdf_reader.getPage(page)
                text += page_obj.extractText()
            return text

    def extract_text_from_docx(file_path):
        return docx2txt.process(file_path)

    def parse_resume(file_path):
        _, ext = os.path.splitext(file_path)
        if ext == '.pdf':
            text = extract_text_from_pdf(file_path)
        elif ext == '.docx':
            text = extract_text_from_docx(file_path)
        else:
            print("Unsupported file format")
            return None
        return text

    def remove_duplicate_sentences(sentences):
        """
        Remove sentences which match 50% or more with other sentences.
        """
        filtered_sentences = []
        for i, sentence in enumerate(sentences):
            is_duplicate = False
            for j in range(i+1, len(sentences)):
                similarity = SequenceMatcher(None, sentence, sentences[j]).ratio()
                if similarity >= 0.5:
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered_sentences.append(sentence)
        return filtered_sentences

    def prepare_bert_input(sentences, seq_len=128, bert_name='bert-base-uncased'):
        tokenizer = BertTokenizer.from_pretrained(bert_name)
        encodings = tokenizer(sentences.tolist(), truncation=True, padding='max_length',
                                    max_length=seq_len)
        input = [np.array(encodings["input_ids"]), np.array(encodings["token_type_ids"]),
                np.array(encodings["attention_mask"])]
        return input


    axes = ["I-E","N-S","T-F","J-P"]
    classes = {"I":0, "E":1, # axis 1
            "N":0,"S":1, # axis 2
            "T":0, "F":1, # axis 3
            "J":0,"P":1} # axis 4
    try:
        parsed_text = parse_resume(cv_path)
        tokenized_data = re.findall(r'\b[\w\s\',-]+\.[\s]*', parsed_text)
        tokenized_data = [re.sub(r"\n", " ", x) for x in tokenized_data]
        filtered_sentences = list(filter(lambda x: len(re.findall(r'\w+', x)) >= 4, tokenized_data))
        filtered_sentences=remove_duplicate_sentences(filtered_sentences)
        processed_text = " ".join(filtered_sentences)
        processed_text = re.sub('\s+', ' ', processed_text).strip()

        sentences = np.asarray([processed_text])
        enc_sentences = prepare_bert_input(sentences)

        opt = tfa.optimizers.RectifiedAdam(learning_rate=3e-5)
        mbtiModel=tf.keras.models.load_model('MBTImodel.h5',custom_objects={"TFBertModel": TFBertModel, "RectifiedAdam": opt})
        predictions = mbtiModel.predict(enc_sentences)
        for sentence, pred in zip(sentences, predictions):
            pred_axis = []
            mask = (pred > 0.5).astype(bool)
            for i in range(len(mask)):
                if mask[i]:
                    pred_axis.append(axes[i][2])
                else:
                    pred_axis.append(axes[i][0])
            print('-- comment: '+sentence.replace("\n", "").strip() +
                '\n-- personality: '+str(pred_axis) +
                '\n-- scores:'+str(pred))
            mbtiScore = {}
            for i in range(len(pred)):
                mbtiScore[pred_axis[i]]=pred[i]
            return mbtiScore
    except Exception as e:
        print("Error occurred: ", e)


def prediction_result(top, aplcnt_name, cv_path, personality_values):
    "after applying a job"
    top.withdraw()
    applicant_data={"Candidate Name":aplcnt_name.get(),  "CV Location":cv_path}
    
    age = personality_values[1]
    
    print("\n############# Candidate Entered Data #############\n")
    print(applicant_data, personality_values)
    
    personality = model.test(personality_values)
    print("\n############# Predicted Personality #############\n")
    print(personality)
    data = ResumeParser(cv_path).get_extracted_data()
    print(data)
    
    try:
        del data['name']
        if len(data['mobile_number'])<10:
            del data['mobile_number']
    except:
        pass
    
    print("\n############# Resume Parsed Data #############\n")

    for key in data.keys():
        if data[key] is not None:
            print('{} : {}'.format(key,data[key]))

    mbtiScore = getMBTIscore(cv_path)
    
    result=Tk()
  #  result.geometry('700x550')
    result.overrideredirect(False)
    result.geometry("{0}x{1}+0+0".format(result.winfo_screenwidth(), result.winfo_screenheight()))
    result.configure(background='White')
    result.title("Predicted Personality")
    
    #Title
    titleFont = font.Font(family='Arial', size=40, weight='bold')
    Label(result, text="Result - Personality Prediction", foreground='green', bg='white', font=titleFont, pady=10, anchor=CENTER).pack(fill=BOTH)
    
    Label(result, text = str('{} : {}'.format("Name:", aplcnt_name.get())).title(), foreground='black', bg='white', anchor='w').pack(fill=BOTH)
    Label(result, text = str('{} : {}'.format("Age:", age)), foreground='black', bg='white', anchor='w').pack(fill=BOTH)
    for key in data.keys():
        if data[key] is not None:
            Label(result, text = str('{} : {}'.format(check_type(key.title()),check_type(data[key]))), foreground='black', bg='white', anchor='w', width=60).pack(fill=BOTH)
    Label(result, text = str("Perdicted Personality: "+personality).title(), foreground='black', bg='white', anchor='w').pack(fill=BOTH)
    Label(result, text = str(f"MBTI Score: {mbtiScore}").title(), foreground='black', bg='white', anchor='w').pack(fill=BOTH)
    
    link = Label(result, text="16 MBTI Personalities",font=('Helveticabold', 15), fg="blue",bg='white', anchor='w', cursor="hand2")
    link.pack()
    link.bind("<Button-1>", lambda e:webbrowser.open_new_tab("https://www.truity.com/page/16-personality-types-myers-briggs"))

    quitBtn = Button(result, text="Exit", command =lambda:  result.destroy()).pack()

    result.mainloop()
    

def perdict_person():
    """Predict Personality"""
    
    # Closing The Previous Window
    root.withdraw()
    
    # Creating new window
    top = Toplevel()
    top.geometry('700x500')
    top.configure(background='black')
    top.title("Apply For A Job")
    
    #Title
    titleFont = font.Font(family='Helvetica', size=20, weight='bold')
    lab=Label(top, text="Personality Prediction", foreground='red', bg='black', font=titleFont, pady=10).pack()

    #Job_Form
    job_list=('Select Job', '101-Developer at TTC', '102-Chef at Taj', '103-Professor at MIT')
    job = StringVar(top)
    job.set(job_list[0])

    l1=Label(top, text="Applicant Name", foreground='white', bg='black').place(x=70, y=130)
    l2=Label(top, text="Age", foreground='white', bg='black').place(x=70, y=160)
    l3=Label(top, text="Gender", foreground='white', bg='black').place(x=70, y=190)
    l4=Label(top, text="Upload Resume", foreground='white', bg='black').place(x=70, y=220)
    l5=Label(top, text="Enjoy New Experience or thing(Openness)", foreground='white', bg='black').place(x=70, y=250)
    l6=Label(top, text="How Offen You Feel Negativity(Neuroticism)", foreground='white', bg='black').place(x=70, y=280)
    l7=Label(top, text="Wishing to do one's work well and thoroughly(Conscientiousness)", foreground='white', bg='black').place(x=70, y=310)
    l8=Label(top, text="How much would you like work with your peers(Agreeableness)", foreground='white', bg='black').place(x=70, y=340)
    l9=Label(top, text="How outgoing and social interaction you like(Extraversion)", foreground='white', bg='black').place(x=70, y=370)
    
    sName=Entry(top)
    sName.place(x=450, y=130, width=160)
    age=Entry(top)
    age.place(x=450, y=160, width=160)
    gender = IntVar()
    R1 = Radiobutton(top, text="Male", variable=gender, value=1, padx=7)
    R1.place(x=450, y=190)
    R2 = Radiobutton(top, text="Female", variable=gender, value=0, padx=3)
    R2.place(x=540, y=190)
    cv=Button(top, text="Select File", command=lambda:  OpenFile(cv))
    cv.place(x=450, y=220, width=160)

    # Create an openness widget (slider)
    openness = Scale(top, from_=1, to=10, orient=HORIZONTAL, length=200,showvalue=0)
    openness.set(5)
    openness.pack()
    openness_label = Label(top, text=openness.get())
    openness_label.place(x=620, y=250)
    openness.config(command=lambda value: openness_label.config(text=value))
    openness.place(x=450, y=250, width=160)

    # Create a neuroticism widget (slider)
    neuroticism = Scale(top, from_=1, to=10, orient=HORIZONTAL, length=200, showvalue=0)
    neuroticism.pack()
    neuroticism.set(5)
    neuroticism_label = Label(top, text=neuroticism.get())
    neuroticism_label.place(x=620, y=280)
    neuroticism.config(command=lambda value: neuroticism_label.config(text=value))
    neuroticism.place(x=450, y=280, width=160, height=20)

    # Create a conscientiousness widget (slider)
    conscientiousness = Scale(top, from_=1, to=10, orient=HORIZONTAL, length=200,showvalue=0)
    conscientiousness.pack()
    conscientiousness.set(5)
    conscientiousness_label = Label(top, text=conscientiousness.get())
    conscientiousness_label.place(x=620, y=310)
    conscientiousness.config(command=lambda value: conscientiousness_label.config(text=value))
    conscientiousness.place(x=450, y=310, width=160)

    # Create an agreeableness widget (slider)
    agreeableness = Scale(top, from_=1, to=10, orient=HORIZONTAL, length=200, showvalue=0)
    agreeableness.pack()
    agreeableness.set(5)
    agreeableness_label = Label(top, text=agreeableness.get())
    agreeableness_label.place(x=620, y=340)
    agreeableness.config(command=lambda value: agreeableness_label.config(text=value))
    agreeableness.place(x=450, y=340, width=160, height=20)

    # Create an extraversion widget (slider)
    extraversion = Scale(top, from_=1, to=10, orient=HORIZONTAL, length=200, showvalue=0)
    extraversion.pack()
    extraversion.set(5)
    extraversion_label = Label(top, text=extraversion.get())
    extraversion_label.place(x=620, y=370)
    extraversion.config(command=lambda value: extraversion_label.config(text=value))
    extraversion.place(x=450, y=370, width=160, height=20)

    submitBtn=Button(top, padx=2, pady=0, text="Submit", bd=0, foreground='white', bg='red', font=(12))
    submitBtn.config(command=lambda: prediction_result(top,sName,loc,(gender.get(),age.get(),openness.get(),neuroticism.get(),conscientiousness.get(),agreeableness.get(),extraversion.get())))
    submitBtn.place(x=350, y=400, width=200)
    

    top.mainloop()

def OpenFile(b4):
    global loc
    name = filedialog.askopenfilename(initialdir="D:/CVdata",
                            filetypes =(("Document","*.docx*"),("PDF","*.pdf*"),('All files', '*')),
                           title = "Choose a file."
                           )
    try:
        filename=os.path.basename(name)
        loc=name
    except:
        filename=name
        loc=name
    b4.config(text=filename)
    return



if __name__ == "__main__":
    model = train_model()
    model.train()

    root = Tk()
    root.geometry('700x500')
    root.configure(background='white')
    root.title("Personality Prediction System")
    titleFont = font.Font(family='Helvetica', size=25, weight='bold')
    homeBtnFont = font.Font(size=12, weight='bold')
    lab=Label(root, text="Personality Prediction System", bg='white', font=titleFont, pady=30).pack()
    b2=Button(root, padx=4, pady=4, width=30, text="Predict Personality", bg='black', foreground='white', bd=1, font=homeBtnFont, command=perdict_person).place(relx=0.5, rely=0.5, anchor=CENTER)
    root.mainloop()
