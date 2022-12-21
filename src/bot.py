from telegram.ext import Updater, MessageHandler, Filters, CommandHandler, ConversationHandler, CallbackQueryHandler
import logging
from token import TOKEN
from fast_search import search_meta

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG
)

logger = logging.getLogger(__name__)


def start(update, context):
    update.message.reply_text("Hi!")
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
    update.message.reply_text(search_meta(query))


def main():
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))
    dp.add_handler(CommandHandler("music", music))
    dp.add_handler(MessageHandler(Filters.text, write_message))
    dp.add_error_handler(error)
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
