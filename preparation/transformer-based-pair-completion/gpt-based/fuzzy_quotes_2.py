import argparse
import json
import random
from typing import List

from api_requests import RequestRunner


MODEL = "gpt-4o"



def get_prompt():
    return (
        "Considering the surrounding context, embed the sentence—either verbatim "
        "or with minimal changes—into a longer, well-formed sentence. "
        "Refer to the following examples as a guide:\n\n"
        "1.\n"
        "Input:\n"
        "Die autem tertio, eleuatis oculis, uidit locum procul, Dixitque ad pueros suos\n"
        "Reused in context:\n"
        "Die autem tertio eleuatis oculis, uidit locum procul, dixitque ad puerum suum, "
        "Expectate hic cum asino, ego et puer illuc usque properantes, postquam "
        "adorauerimus, reuertemur ad uos.\n\n"
        "2.\n"
        "Input:\n"
        "ego autem et puer illuc usque properantes, postquam adorauerimus, "
        "reuertemur ad uos.\n"
        "Reused in context:\n"
        "Die autem tertio eleuatis oculis, uidit locum procul, dixitque ad puerum suum, "
        "Expectate hic cum asino, ego et puer illuc usque properantes, postquam "
        "adorauerimus, reuertemur ad uos.\n\n"
        "3.\n"
        "Input:\n"
        "Dixitque ei, Non extendas manum tuam super puerum, neque facias illi quicquam\n"
        "Reused in context:\n"
        "Ebraica magis sunt elegantia, Ne extendas manum tuam super puerum, nec feceris "
        "ei quicquam, quoniam nunc cognoui quod times tu deum, quod non pepercisti "
        "filio tuo unico a me.\n\n"
        "4.\n"
        "Input:\n"
        "nunc cognoui quod timeas Dominum, et non pepercisti unigenito filio tuo propter ne.\n"
        "Reused in context:\n"
        "Ebraica magis sunt elegantia, Ne extendas manum tuam super puerum, nec feceris "
        "ei quicquam, quoniam nunc cognoui quod times tu deum, quod non pepercisti "
        "filio tuo unico a me.\n\n"
        "5.\n"
        "Input:\n"
        "Isaac typus est Christi, qui uerum animi gaudium est, uera pax, uera tranquillitas, "
        "et si autem passio Christi tempestiue promissa sit, tamen tertio primum die Christus "
        "ad locum passionis uenit.\n"
        "Reused in context:\n"
        "Isaac typus est Christi, qui uerum animi gaudium est, uera pax, uera tranquillitas.\n\n"
        "6.\n"
        "Input:\n"
        "Quod etiam ex responsione Abrahae ad filium clarum fit, cum dicit\n"
        "Reused in context:\n"
        "Quod etiam ex responsione Abrahae ad filium clarum fit, cum dicit, Dominus "
        "uidebit de uictima fili mi.\n\n"
        "7.\n"
        "Input:\n"
        "Dominus uidebit, hoc est exitum ostendet, liberabit, aderit, opem ostendet.\n"
        "Reused in context:\n"
        "Hinc nimirum factum est, ut deinde in quacunque ardua et dubia re fideles "
        "dicerent, Dominus uidebit, hoc est, exitum ostendet, liberabit, aderit, opem feret.\n\n"
        "8.\n"
        "Input:\n"
        "Ubi solius Dei auxilium et imploras et expectas,\n"
        "Reused in context:\n"
        "haec demum est tentatio, ubi solius dei auxilium et imploras et expectas.\n\n"
        "9.\n"
        "Input:\n"
        "Ubi solius Dei auxilium et imploras et expectas,\n"
        "Reused in context:\n"
        "Tum repente aderit dominus, et emediis procellis educet in se sperantes.\n"
    )


def create_fuzzy_quotes_2(
    input_file: str,
    output_file: str,
    sample_size: int,
    batch_size: int
):

    fuzzy_quoter = RequestRunner(MODEL, "Latin")
    batches = fuzzy_quoter.load(input_file, sample_size, batch_size)

    result = []
    for batch in batches:
        fuzzy_quotes = fuzzy_quoter.process(
            batch,
            random.uniform(0.8, 1.2),
            get_prompt()
        )

        for (i, sent1), (j, sent2) in zip(batch, fuzzy_quotes):
            if i == j:
                result.append((sent1, sent2))

    fuzzy_quoter.save(result, "fuzzy_quote", output_file)




def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    parser.add_argument("sample_size", type=int)
    parser.add_argument("batch_size", type=int)
    args = parser.parse_args()

    create_fuzzy_quotes_2(
        args.input_file, 
        args.output_file, 
        args.sample_size, 
        args.batch_size
    )

if __name__ == "__main__":
    main()

