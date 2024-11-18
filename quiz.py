import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_words(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file]

def vectorize_words(words):
    vectorizer = CountVectorizer(analyzer='char', lowercase=False)
    return vectorizer, vectorizer.fit_transform(words)

def calculate_similarity(vec1, vec2):
    return cosine_similarity(vec1, vec2)[0][0]

def get_hint(word):
    return f"힌트: 첫 글자는 '{word[0]}' 입니다."

def play_quiz(words, vectors, vectorizer):
    target_word = np.random.choice(words)
    target_vector = vectors[words.index(target_word)]
    
    print("단어를 맞춰보세요!")
    print(get_hint(target_word))
    
    while True:
        guess = input("추측한 단어를 입력하세요 (포기하려면 '포기'를 입력하세요): ")
        if guess.lower() == '포기':
            print(f"정답은 '{target_word}'입니다.")
            break
        if guess == target_word:
            print("정답입니다!")
            break
        
        guess_vector = vectorizer.transform([guess])
        similarity = calculate_similarity(target_vector, guess_vector)
        
        if guess not in words:
            print(f"'{guess}'는 단어 목록에 없습니다.")
        
        print(f"입력한 단어: {guess}")
        print(f"유사도: {similarity:.4f}")

def main():
    words = load_words('word.txt')
    vectorizer, vectors = vectorize_words(words)
    
    while True:
        play_quiz(words, vectors, vectorizer)
        if input("계속하시겠습니까? (y/n): ").lower() != 'y':
            break

if __name__ == "__main__":
    main()