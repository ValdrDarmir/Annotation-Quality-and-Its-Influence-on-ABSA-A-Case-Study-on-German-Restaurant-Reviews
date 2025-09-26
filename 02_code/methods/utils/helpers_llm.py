PROMPT_TASD = """Entsprechend der folgenden Definition der Sentiment-Elemente:

Der "Aspektbegriff" bezieht sich auf ein bestimmtes Merkmal, eine Eigenschaft oder einen Aspekt eines Produkts oder einer Dienstleistung, zu dem eine Person eine Meinung äußern kann. Explizite Aspektbegriffe kommen explizit als Teilzeichenkette im gegebenen Text vor. Der Aspektbegriff kann "null" sein, wenn es sich um einen impliziten Aspekt handelt.

Die "Aspektkategorie" bezeichnet die Kategorie, zu der der Aspekt gehört. Die verfügbaren Kategorien sind: {categories}.

Die 'Sentiment-Polarität' beschreibt den Grad der Positivität, Negativität oder Neutralität, die in der Meinung zu einem bestimmten Aspekt oder Merkmal eines Produkts oder einer Dienstleistung ausgedrückt wird. Die verfügbaren Polaritäten sind: 'positive', 'negative' und 'neutral'.

Bitte befolge die Anweisungen sorgfältig. Stelle sicher, dass Aspektbegriffe exakt mit dem Text übereinstimmen oder "null" sind, wenn sie implizit sind. Stelle sicher, dass Aspektkategorien aus den angegebenen Kategorien stammen. Stelle sicher, dass die Polaritäten aus den verfügbaren Polaritäten stammen.

Erkenne alle Sentiment-Elemente mit ihren jeweiligen Aspektbegriffen, Aspektkategorien und Sentiment-Polaritäten im folgenden Text im Format [('Aspektbegriff', 'Aspektkategorie', 'Sentiment-Polarität'), ...]:

{examples}"""

PROMPT_ACSA = """Entsprechend der folgenden Definition der Sentiment-Elemente:

Die "Aspektkategorie" bezeichnet die Kategorie, zu der der Aspekt gehört. Die verfügbaren Kategorien sind: {categories}.

Die 'Sentiment-Polarität' beschreibt den Grad der Positivität, Negativität oder Neutralität, die in der Meinung zu einem bestimmten Aspekt oder Merkmal eines Produkts oder einer Dienstleistung ausgedrückt wird. Die verfügbaren Polaritäten sind: 'positive', 'negative' und 'neutral'.

Bitte befolge die Anweisungen sorgfältig. Stelle sicher, dass Aspektkategorien aus den angegebenen Kategorien stammen. Stelle sicher, dass die Polaritäten aus den verfügbaren Polaritäten stammen.

Erkenne alle Sentiment-Elemente mit ihren jeweiligen Aspektkategorien und Sentiment-Polaritäten im folgenden Text im Format [('Aspektkategorie', 'Sentiment-Polarität'), ...]:

{examples}"""