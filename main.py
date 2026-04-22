import asyncio
import json
import logging
import os
import re
import time
from urllib import error, request

from aiohttp import web
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import KeyboardButton, Message, ReplyKeyboardMarkup
from dotenv import load_dotenv

load_dotenv()


def read_env(name: str, default: str = "") -> str:
	return os.getenv(name, default).strip().strip('"').strip("'")


BOT_TOKEN = read_env("BOT_TOKEN")
GEMINI_API_KEY = read_env("GEMINI_API_KEY") or read_env("GOOGLE_API_KEY")
GEMINI_COOLDOWN_UNTIL = 0.0
DEFAULT_LANGUAGE = "english"
SUPPORTED_LANGUAGES = {"English": "english", "Kazakh": "kazakh", "Russian": "russian"}
TELEGRAM_MESSAGE_LIMIT = 4000
HEALTH_PORT = int(read_env("PORT", "10000"))

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

language_keyboard = ReplyKeyboardMarkup(
	keyboard=[
		[KeyboardButton(text="English")],
		[KeyboardButton(text="Kazakh")],
		[KeyboardButton(text="Russian")],
	],
	resize_keyboard=True,
)

user_languages: dict[int, str] = {}

qa_english = {
	"algorithm": "An algorithm is a step-by-step method to solve a problem.",
	"variable": "A variable is a named storage for data in a program.",
	"loop": "A loop repeats a block of code while a condition is true.",
}

qa_kazakh = {
	"алгоритм": "Алгоритм - есепті шешудің қадамдық тәсілі.",
	"айнымалы": "Айнымалы - бағдарламадағы деректерді сақтайтын атаулы орын.",
	"цикл": "Цикл - шарт орындалғанша код блогын қайталайды.",
}

qa_russian = {
	"алгоритм": "Алгоритм - это пошаговый способ решения задачи.",
	"переменная": "Переменная - это именованное место для хранения данных.",
	"цикл": "Цикл повторяет блок кода, пока условие истинно.",
}
CS_KEYWORDS = {
	"algorithm", "data", "database", "network", "protocol", "http", "https", "tcp", "udp", "ip", "dns",
	"programming", "code", "class", "object", "oop", "polymorphism", "inheritance", "encapsulation",
	"abstraction", "function", "variable", "loop", "array", "list", "stack", "queue", "tree", "graph",
	"binary", "recursion", "sorting", "search", "complexity", "ai", "machine learning", "neural", "cybersecurity",
	"python", "java", "javascript", "c++", "api", "server", "client", "os", "linux", "windows", "compiler",
	"информатика", "алгоритм", "переменная", "цикл", "массив", "граф", "дерево", "стек", "очередь", "рекурсия", "структура данных", "структуры данных",
	"полиморфизм", "наследование", "инкапсуляция", "абстракция", "протокол", "сеть", "база данных", "программирование",
	"кибербезопасность", "жасанды интеллект", "алгоритм", "айнымалы", "цикл", "граф", "ағаш", "рекурсия", "деректер құрылымы", "деректер құрылымдары",
	"полиморфизм", "мұрагерлік", "инкапсуляция", "абстракция", "желі", "дерекқор", "бағдарламалау",
}


def get_dictionary_by_language(language: str) -> dict[str, str]:
	if language == "kazakh":
		return qa_kazakh
	if language == "russian":
		return qa_russian
	return qa_english


def parse_retry_seconds(error_body: str) -> int:
	retry_match = re.search(r'"retryDelay"\s*:\s*"(\d+)s"', error_body)
	return int(retry_match.group(1)) if retry_match else 60


def ask_gemini_cs_sync(user_text: str, language: str) -> str | None:
	global GEMINI_COOLDOWN_UNTIL

	if not GEMINI_API_KEY:
		return None

	if time.time() < GEMINI_COOLDOWN_UNTIL:
		return None

	url = (
		"https://generativelanguage.googleapis.com/v1beta/models/"
		"gemini-3-flash-preview:generateContent?key="
		f"{GEMINI_API_KEY}"
	)

	prompt = (
		"You are a computer science tutor. "
		"If the user question is NOT about computer science, reply exactly: No idea. "
		"If it IS about computer science, provide a complete answer in 2-4 sentences and finish the thought. "
		"Do not cut off mid-sentence. Do not use bullet points unless the user asks for them. "
		f"Reply in the selected language: {language}. "
		f"User question: {user_text}"
	)

	payload = {
		"contents": [{"parts": [{"text": prompt}]}],
		"generationConfig": {"temperature": 0.2, "maxOutputTokens": 1000},
	}

	data = json.dumps(payload).encode("utf-8")
	req = request.Request(
		url=url,
		data=data,
		headers={"Content-Type": "application/json"},
		method="POST",
	)

	try:
		with request.urlopen(req, timeout=15) as resp:
			body = json.loads(resp.read().decode("utf-8"))
			candidates = body.get("candidates", [])
			if not candidates:
				logging.warning("Gemini response has no candidates: %s", body)
				return None
			parts = candidates[0].get("content", {}).get("parts", [])
			if not parts:
				logging.warning("Gemini response has no content parts: %s", body)
				return None
			text = "\n".join(
				part.get("text", "").strip()
				for part in parts
				if isinstance(part, dict) and part.get("text")
			).strip()
			return text if text else None
	except error.HTTPError as exc:
		error_body = ""
		try:
			error_body = exc.read().decode("utf-8", errors="ignore")
		except Exception:
			error_body = ""

		if exc.code == 429:
			retry_seconds = parse_retry_seconds(error_body)
			GEMINI_COOLDOWN_UNTIL = time.time() + retry_seconds
			logging.warning("Gemini quota exceeded. Cooling down for %s seconds.", retry_seconds)
			return None

		logging.warning("Gemini HTTP error %s: %s", exc.code, error_body)
		return None
	except (error.URLError, TimeoutError, json.JSONDecodeError):
		logging.exception("Gemini request failed")
		return None


async def ask_gemini_cs(user_text: str, language: str) -> str | None:
	return await asyncio.to_thread(ask_gemini_cs_sync, user_text, language)


def is_cs_related(question: str) -> bool:
	normalized = question.lower()
	return any(keyword in normalized for keyword in CS_KEYWORDS)


def clean_response_text(text: str) -> str:
	return " ".join(text.split()).strip()


async def send_clean_message(message: Message, text: str) -> None:
	clean_text = clean_response_text(text)
	if not clean_text:
		return

	for start in range(0, len(clean_text), TELEGRAM_MESSAGE_LIMIT):
		await message.answer(clean_text[start : start + TELEGRAM_MESSAGE_LIMIT])


@dp.message(CommandStart())
async def cmd_start(message: Message) -> None:
	user_name = message.from_user.first_name if message.from_user else "User"
	if message.from_user:
		user_languages[message.from_user.id] = DEFAULT_LANGUAGE
	await message.answer(
		f"Hello, {user_name}\nChoose language:",
		reply_markup=language_keyboard,
	)


@dp.message(F.text.in_({"English", "Kazakh", "Russian"}))
async def choose_language(message: Message) -> None:
	if not message.from_user:
		return

	selected = SUPPORTED_LANGUAGES.get(message.text, DEFAULT_LANGUAGE)
	user_languages[message.from_user.id] = selected
	await message.answer(f"Language set to {message.text}. Now send a CS keyword.")


@dp.message(F.text)
async def answer_cs_questions(message: Message) -> None:
	if not message.from_user or not message.text:
		return

	question = message.text.strip()
	keyword = question.lower()
	language = user_languages.get(message.from_user.id, DEFAULT_LANGUAGE)
	answer = get_dictionary_by_language(language).get(keyword)

	if answer:
		await send_clean_message(message, answer)
		return
	if not is_cs_related(question):
		await message.answer("No idea")
		return

	gemini_answer = await ask_gemini_cs(question, language)
	if gemini_answer:
		await send_clean_message(message, gemini_answer)
		return

	if time.time() < GEMINI_COOLDOWN_UNTIL:
		await message.answer("Gemini quota is exceeded now. Please try again in a minute.")
		return

	await message.answer("Gemini is unavailable right now. Try again later.")


async def main() -> None:
	if not BOT_TOKEN:
		raise ValueError("Set BOT_TOKEN in environment variables before running.")

	app = web.Application()

	async def healthz(_: web.Request) -> web.Response:
		return web.json_response({"status": "ok"})

	app.router.add_get("/healthz", healthz)
	runner = web.AppRunner(app)
	await runner.setup()
	site = web.TCPSite(runner, host="0.0.0.0", port=HEALTH_PORT)
	await site.start()
	logging.info("Health server is running on 0.0.0.0:%s/healthz", HEALTH_PORT)

	try:
		await dp.start_polling(bot)
	finally:
		await runner.cleanup()


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO)
	try:
		asyncio.run(main())
	except KeyboardInterrupt:
		logging.info("Bot stopped by user.")

