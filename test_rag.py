from src.llm.chains import ask_bot
from src.logger import get_logger

question = "З якими backend технологіями працює компанія?"
answer = ask_bot(question)

logger = get_logger(__name__)

logger.info(f"\n❓ Питання: {question}")
logger.info(f"\n🤖 Відповідь:\n{answer}")