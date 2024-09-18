import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import matplotlib.pyplot as plt
# Descargas necesarias para nltk
nltk.download('punkt_tab')
nltk.download('wordnet')

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import matplotlib.pyplot as plt

# Descargas necesarias para nltk
nltk.download('punkt')  # Corrección aquí: 'punkt' en lugar de 'punkt_tab'
nltk.download('wordnet')
nltk.download('omw-1.4')

# Texto de ejemplo
text = "El análisis de textos es fundamental para extraer información valiosa."

# Tokenización
tokens = word_tokenize(text)
print("Tokens:", tokens)

# Lematización
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(token) for token in tokens]
print("Lematizados:", lemmatized_words)

# Análisis de frecuencia de palabras
word_freq = pd.Series(lemmatized_words).value_counts()
print("Frecuencia de palabras:\n", word_freq)

# Visualización
plt.figure(figsize=(10, 6))
word_freq.plot(kind='bar', color='skyblue')
plt.title('Frecuencia de Palabras Lematizadas')
plt.xlabel('Palabras')
plt.ylabel('Frecuencia')
plt.xticks(rotation=45)
plt.show()