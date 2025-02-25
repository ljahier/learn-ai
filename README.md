# Learn AI

Dataset :

- La classe (un signe "+" pour promoteur ou "-" pour non-promoteur)
- Le nom de l'instance
- La séquence complète de 57 nucléotides

Encodage One-Hot, chaque nucléotide (a, g, t, c) est transformé en un vecteur de 4 valeurs. La séquence entière est convertie en un vecteur de 228 valeurs.

Modèle de Régression Logistique https://en.wikipedia.org/wiki/Logistic_regression

Le modèle calcule la somme pondérée des entrées plus un biais, puis applique une fonction sigmoïde pour produire une probabilité. La prédiction est proche de 1 pour un promoteur et proche de 0 pour un non-promoteur.

Entraînement par Descente de Gradient https://fr.wikipedia.org/wiki/Algorithme_du_gradient
Le modèle est entraîné sur l'ensemble du dataset pendant un nombre d'epochs défini. Les poids et le biais sont ajustés pour réduire l'erreur moyenne sur les prédictions.

L'erreur moyenne est affichée périodiquement durant l'entraînement. A la fin le programme affiche le biais, les premiers poids et la prédiction pour chaque instance du dataset.
