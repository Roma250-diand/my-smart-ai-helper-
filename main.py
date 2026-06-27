from flask import Flask, render_template, request
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import re

app = Flask(__name__)

# Загружаем модели
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model='blanchefort/rubert-base-cased-sentiment'
)
tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/rugpt3medium_based_on_gpt2")
model = AutoModelForCausalLM.from_pretrained("sberbank-ai/rugpt3medium_based_on_gpt2")

# Словарь жанров и ключевых слов для распознавания предпочтений
GENRE_KEYWORDS = {
    "экшен": ["экшен", "боевик", "шутер", "стрельба", "бои", "драка", "action", "shooter", "бой"],
    "ролевая": ["rpg", "ролевая", "ролев", "персонаж", "уровень", "прокачка", "квесты", "диалоги"],
    "стратегия": ["стратегия", "тактика", "стратег", "командир", "армия", "война", "постройка", "база"],
    "симулятор": ["симулятор", "сим", "реализм", "жизнь", "ферма", "город", "управление", "строительство"],
    "головоломка": ["головоломка", "пазл", "логика", "загадка", " puzzle", "brain", "мышление"],
    "хоррор": ["хоррор", "ужасы", "страх", "horror", "жутко", "монстры", "выживание", "dark"],
    "приключения": ["приключение", "адвенчура", "adventure", "путешествие", "исследование", "открытый мир"],
    "гонки": ["гонки", "авто", "машина", "скорость", "трасса", "racing", "drive"],
    "спорт": ["спорт", "футбол", "баскетбол", "хоккей", "теннис", "sport", "олимпиада"],
    "фэнтези": ["фэнтези", "магия", "эльфы", "дракон", "меч", "fantasy", "маги"],
    "научная фантастика": ["научная фантастика", "фантастика", "космос", "киберпанк", "робот", "scifi", "future"],
    "музыка": ["музыка", "ритм", "танцы", "музыкальный", "ноты", "rock", "rhythm"],
    "аркада": ["аркада", "платформер", "классика", "ретро", "arcade", "платформа"],
    "выживание": ["выживание", "выживать", "ресурсы", "крафт", "постройка", "база", "голод", "жажда", "survival"],
    "песочница": ["песочница", "свобода", "творчество", "sandbox", "строить", "разрушать"],
    "stealth": ["стелс", "скрытность", "невидимка", "тихо", "убийца", "скрываться", "незаметно"],
    "ммо": ["ммо", "мморпг", "онлайн", "многопользовательский", "world of warcraft", "wow", "массовый"],
    "инди": ["инди", "независимый", "small", "минимализм", "атмосфера", "инди-игра"]
}

# Обратный словарь: жанр -> русское название (для красивого вывода)
GENRE_NAMES = {
    "экшен": "экшен",
    "ролевая": "ролевая игра (RPG)",
    "стратегия": "стратегия",
    "симулятор": "симулятор",
    "головоломка": "головоломка",
    "хоррор": "хоррор",
    "приключения": "приключения",
    "гонки": "гонки",
    "спорт": "спорт",
    "фэнтези": "фэнтези",
    "научная фантастика": "научная фантастика",
    "музыка": "музыкальная игра",
    "аркада": "аркада",
    "выживание": "выживание",
    "песочница": "песочница",
    "stealth": "стелс",
    "ммо": "MMO",
    "инди": "инди-игра"
}


def detect_genre(text):
    """Определяет жанр по тексту пользователя."""
    text_lower = text.lower()
    detected = []

    for genre, keywords in GENRE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                detected.append(genre)
                break

    detected = list(dict.fromkeys(detected))

    if not detected:
        return "неопределенный жанр", []

    return detected[0], detected


def generate_multiple_recommendations(genre, user_text, count=4):
    """
    Генерирует список из нескольких игр на основе жанра и текста пользователя.
    """
    genre_name = GENRE_NAMES.get(genre, genre)

    # Формируем промпт для получения списка игр
    prompt = (
        f"Посоветуй {count} самых популярных и лучших игр в жанре {genre_name}. "
        f"Игры должны подходить для человека, который сказал: \"{user_text[:100]}\". "
        f"Напиши список названий игр через запятую, без нумерации, без описаний, без лишних слов. "
        f"Только названия игр."
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        top_p=0.85,
        temperature=0.7,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    result = text[len(prompt):].strip()

    # Очищаем результат
    result = re.sub(r'^["\']|["\']$', '', result)

    # Разбиваем на отдельные игры (по запятым, точкам с запятой или переводу строки)
    games = re.split(r'[,;.\n]\s*', result)
    games = [g.strip() for g in games if g.strip()]

    # Убираем возможную нумерацию в начале
    cleaned_games = []
    for game in games:
        # Убираем цифры и точки в начале (1., 2. и т.д.)
        cleaned = re.sub(r'^\d+[\.\)]\s*', '', game)
        cleaned = re.sub(r'^[-\*]\s*', '', cleaned)
        if cleaned and len(cleaned) > 1:
            cleaned_games.append(cleaned)

    # Если игр меньше, чем запрошено, добавляем еще
    if len(cleaned_games) < count:
        # Пробуем получить дополнительный список
        extra_prompt = (
            f"Еще {count - len(cleaned_games)} популярных игры в жанре {genre_name}. "
            f"Только названия через запятую."
        )
        inputs = tokenizer(extra_prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=True,
            top_p=0.85,
            temperature=0.8,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
        extra_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        extra_result = extra_text[len(extra_prompt):].strip()
        extra_games = re.split(r'[,;.\n]\s*', extra_result)
        extra_games = [g.strip() for g in extra_games if g.strip()]

        for g in extra_games[:count - len(cleaned_games)]:
            cleaned = re.sub(r'^\d+[\.\)]\s*', '', g)
            cleaned = re.sub(r'^[-\*]\s*', '', cleaned)
            if cleaned and len(cleaned) > 1:
                cleaned_games.append(cleaned)

    # Ограничиваем количество и возвращаем
    return cleaned_games[:count], genre_name


@app.route("/", methods=["GET", "POST"])
def index():
    recommendation = ""
    user_text = ""

    if request.method == "POST":
        user_text = request.form["message"].strip()

        if user_text:
            genre, all_genres = detect_genre(user_text)

            if genre == "неопределенный жанр":
                # Если жанр не определен, используем общий запрос
                prompt = (
                    f"Посоветуй 4 популярные игры для человека с такими предпочтениями: {user_text[:80]}. "
                    f"Ответь списком названий через запятую."
                )
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=80,
                    do_sample=True,
                    top_p=0.85,
                    temperature=0.7,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id
                )
                ai_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                ai_text = ai_text[len(prompt):].strip()
                ai_text = re.sub(r'^["\']|["\']$', '', ai_text)

                # Разбиваем на игры
                games = re.split(r'[,;.\n]\s*', ai_text)
                games = [g.strip() for g in games if g.strip()]
                games = [re.sub(r'^\d+[\.\)]\s*', '', g) for g in games]
                games = [re.sub(r'^[-\*]\s*', '', g) for g in games if g]
                games = games[:4]

                if games:
                    games_list = "<br>".join([f"{i + 1}. {game}" for i, game in enumerate(games)])
                    recommendation = f'Ваше предпочтение: {user_text[:100]}.<br><br>Рекомендуемые игры:<br><strong>{games_list}</strong>'
                else:
                    recommendation = "Не удалось сгенерировать рекомендации. Попробуйте переформулировать запрос."
            else:
                # Генерируем несколько рекомендаций
                games, genre_name = generate_multiple_recommendations(genre, user_text, count=4)

                if len(all_genres) > 1:
                    additional = f" (найдены жанры: {', '.join([GENRE_NAMES.get(g, g) for g in all_genres])})"
                else:
                    additional = ""

                if games:
                    games_list = "<br>".join([f"{i + 1}. {game}" for i, game in enumerate(games)])
                    recommendation = f'Жанр: <strong>{genre_name}</strong>{additional}.<br><br>Рекомендую попробовать:<br><strong>{games_list}</strong>'
                else:
                    recommendation = "Не удалось сгенерировать рекомендации. Попробуйте переформулировать запрос."
        else:
            recommendation = "Пожалуйста, введите описание ваших предпочтений."

    return render_template("index.html", recommendation=recommendation, user_text=user_text)


if __name__ == "__main__":
    app.run(debug=True)