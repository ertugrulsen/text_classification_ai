import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification,BertTokenizer, BertForSequenceClassification

def load_model():
    #model = DistilBertForSequenceClassification.from_pretrained("models", cache_dir="./cache")  # Model dosyaları ./cache klasörüne indirilecek
    model = DistilBertForSequenceClassification.from_pretrained("./models")
    tokenizer = DistilBertTokenizer.from_pretrained("dbmdz/distilbert-base-turkish-cased")  # Tokenizer dosyaları ./cache klasörüne indirilecek
    return model, tokenizer



def classify_text(text, model, tokenizer):
    encoding = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**encoding)
    logits = outputs.logits
    predicted_class = torch.argmax(logits).item()

    return predicted_class

model, tokenizer = load_model()

sample_texts = [
    "Müşteri hizmetleri çağrı merkezine ulaşmak için karmaşık menüleri takip etmek zorunda kaldım",
    "Kredi başvurusu için gereken belgeleri tamamladım ancak bankanızın belirttiği sürede geri dönüş alamadım",
    "Müşteri hizmetleri temsilciniz hesap kesim tarihimi değiştirmem konusunda yardımcı oldu",
    "Kredi başvurusu sırasında banka yetkilileri güler yüzlü ve yardımsever davrandı",
    "Banka aracılığıyla yaptığım EFT işlemi uzun süreli gecikmelerle karşılaşıyor",
    "Mobil uygulamanızın arayüzü şık ve basit kullanımı oldukça keyifli",
    "Ücretlendirme politikanız müşterilere sağladığınız avantajlarla dengeli ve adil bir şekilde uygulanıyor",
    "ATM'lerinizdeki işlem hızı oldukça hızlı bekleme süresi minimum düzeyde",
    "POS cihazınızın işlem hızı oldukça yüksek sıra beklemeden ödeme yapabiliyorum",
    "Şubenizdeki hizmet kalitesi memnuniyetimi sağlayacak düzeyde ve başarılıydı",
    "Faiz işlemi sırasında hesabımdan fazla faiz kesildi iade talep ediyorum"
]

for text in sample_texts:
    predicted_class = classify_text(text, model, tokenizer)
    label2id = {0: "müşteri hizmetleri",
                1: "kredi başvurusu",
                2: "mobil uygulama",
                3: "ücretlendirme",
                4: "atm",
                5: "kredi kartı",
                6: "internet bankacılığı",
                7: "faiz işlem",
                8: "şube hizmeti",
                9: "çağrı merkezi",
                10: "para transferi",
                11: "pos", }

    predicted_category = label2id[predicted_class]
    print(f"Metin: {text}")
    print(f"Tahmin: {predicted_category}\n")