import argparse
import sys
import os


from .logging_config import get_logger
from .images.part1_download import prepare_json
from .images.processing import process_
from .analysis import aggregations
logger = get_logger(__name__)


def prepare_command(csv_path, output_json):
    logger.info(f"Запуск подготовки JSON из {csv_path}")
    result = prepare_json(csv_path, output_json, limit=10)
    if result:
        logger.info(f"JSON сохранён: {output_json}")
    else:
        logger.error("Не удалось подготовить JSON")
        sys.exit(1)


def process_command(input_json, output_dir, num):
    logger.info(f"Запуск обработки {num} изображений")

    if not os.path.exists(input_json):
        logger.error(f"Файл {input_json} не найден")
        sys.exit(1)

    try:
        result = process_(input_json, output_dir, num)
        logger.info(f"Обработано {result} изображений")
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        sys.exit(1)


def analyze_command(csv_path, output_dir):
    logger.info(f"Запуск анализа датасета {csv_path}")
    aggregations.file_ = csv_path
    aggregations.OUTPUT_DIR = output_dir
    aggregations.main_pipeline()
    logger.info("Анализ завершён")


def main():
    parser = argparse.ArgumentParser(prog="metetl")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # prepare
    p = subparsers.add_parser("prepare")
    p.add_argument("--csv", required=True)
    p.add_argument("--output", default="data/to_download.json")
    def wrapper(args):
        return prepare_command(args.csv, args.output)
    p.set_defaults(func=wrapper)

    # process
    p = subparsers.add_parser("process")
    p.add_argument("--input", default="data/to_download.json")
    p.add_argument("--output-dir", default="images")
    p.add_argument("--num", type=int, default=5)
    def wrapper_process(args):
        return process_command(args.input, args.output_dir, args.num)
    p.set_defaults(func=wrapper_process)

    # analyze
    p = subparsers.add_parser("analyze")
    p.add_argument("--csv", required=True)
    p.add_argument("--output-dir", default="data/plots")
    def wrapper_analyze(args):
        return analyze_command(args.csv, args.output_dir)
    p.set_defaults(func=wrapper_analyze)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()