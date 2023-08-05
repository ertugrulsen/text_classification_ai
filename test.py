import openai

# OpenAI API anahtarınızı buraya girin
api_key = "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"
openai.api_key = api_key

# GPT-3.5 modeline metin girişi yapmak için bir örnek
prompt_text = "Merhaba, ben ChatGPT. Seninle sohbet etmek için buradayım."

def interact_with_chatgpt(prompt):
    try:
        response = openai.Completion.create(
            engine="text-davinci-002",  # GPT-3.5 modeli
            prompt=prompt,
            temperature=0.7,  # Daha az tahmin edilebilir cevaplar için düşük değerler (0.2'ye yakın), daha rastgele cevaplar için yüksek değerler (1.0)
            max_tokens=150,   # Oluşturulan cevapların maksimum uzunluğu
            stop=["\n"],      # Cevabın otomatik kesilmesini sağlamak için yeni satıra kadar olan kısmı alıyoruz
            n=1,              # Birden fazla cevap almak isterseniz değiştirebilirsiniz
            language="tr"     # Türkçe dil kodu
        )
        reply = response['choices'][0]['text'].strip()
        return reply
    except Exception as e:
        print("Hata oluştu:", str(e))
        return None

# ChatGPT ile etkileşim başlatma
while True:
    user_input = input("Siz: ")
    if user_input.lower() == 'çık':
        print("Chat sonlandırıldı.")
        break

    prompt_text += "\nSen: " + user_input
    response_text = interact_with_chatgpt(prompt_text)
    if response_text:
        print("ChatGPT:", response_text)
        prompt_text += "\nChatGPT:" + response_text
