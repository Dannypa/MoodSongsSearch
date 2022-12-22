import threading

from telegram.ext import Updater, MessageHandler, Filters, CommandHandler, ConversationHandler, CallbackQueryHandler
import logging
from config import TOKEN
import fast_search

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG
)

logger = logging.getLogger(__name__)
updater = None

def start(update, context):
    update.message.reply_text("Привет! \nHi!")
    context.user_data['messages'] = []


def help(update, context):
    update.message.reply_text('''Этот бот может давать музыкальные рекомендации на основе ваших сообщений.
    Для получения рекомендаций введите команду /music
    \n\n This bot can recommend music based on your messages. To get a recommendation type /music''')


def write_message(update, context):
    context.user_data['messages'].append(update.message.text)


def error(update, context):
    logger.warning('Update "%s" caused error "%s"', update, context.error)


def music(update, context):
    query = ' '.join(map(str, context.user_data['messages']))
    context.user_data['messages'] = []
    text = "Вот ваши рекомендации: / " \
           "Here are your recommendations: \n"
    res = fast_search.get_songs(query)
    for song in res:
        text += f'{song[1]} - {song[0]} \n'
    update.message.reply_text(text)


def stop(update, context):
    update.message.reply_text("Приятного прослушивания!")
    threading.Thread(target=shutdown).start()


def main():
    global updater
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))
    dp.add_handler(CommandHandler("music", music))
    dp.add_handler(CommandHandler("stop", stop))
    dp.add_handler(MessageHandler(Filters.text, write_message))
    dp.add_error_handler(error)
    updater.start_polling()
    updater.idle()


def shutdown():
    updater.stop()
    updater.is_idle = False


if __name__ == '__main__':
    main()
