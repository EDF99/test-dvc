from transformers import MT5Tokenizer, MT5ForConditionalGeneration, DataCollatorForSeq2Seq
import pandas as pd
import numpy as np
import evaluate

def custom_eval(predictions, references):
    agregar_contacto_len = 0
    transferencia_len = 0
    total_len = len(predictions)
    correct_action = 0
    correct_alias = 0
    correct_nombre = 0
    correct_monto = 0
    correct_ent = 0
    correct_cuenta = 0
    correct_moneda = 0
    correct_doc = 0
    for prediction, reference in zip(predictions, references):
        splitted_prediction = prediction.split("|")
        len_pred = len(splitted_prediction)
        
        splitted_reference = reference.split("|")
        reference_action = splitted_reference[0]

        if(splitted_prediction[0] == reference_action):
            correct_action += 1

        if(reference_action == "T" or reference_action == "A"):
            # T|alias|name|monto|entidad|nro_cuenta|moneda|nro_doc
            # A|alias|name|monto|entidad|nro_cuenta|moneda|nro_doc
            transferencia_len += 1
            if(len_pred > 1 and splitted_prediction[1].lower() == splitted_reference[1].lower()):
                correct_alias += 1
            if(len_pred > 2 and splitted_prediction[2].lower() == splitted_reference[2].lower()):
                correct_nombre += 1
            if(len_pred > 3 and splitted_prediction[3].lower() == splitted_reference[3].lower()):
                correct_monto += 1
            if(len_pred > 4 and splitted_prediction[4].lower() == splitted_reference[4].lower()):
                correct_ent += 1
            if(len_pred > 5 and splitted_prediction[5].lower() == splitted_reference[5].lower()):
                correct_cuenta += 1
            if(len_pred > 6 and splitted_prediction[6].lower() == splitted_reference[6].lower()):
                correct_moneda += 1
            if(len_pred > 7 and splitted_prediction[7].lower() == splitted_reference[7].lower()):
                correct_doc += 1

    action_acc = correct_action/total_len
    if(transferencia_len == 0 and agregar_contacto_len == 0):
        alias_acc = nombre_acc = cuenta_acc = entidad_acc = doc_acc = monto_acc = moneda_acc = -1
    else:
        alias_acc = (correct_alias/(transferencia_len+agregar_contacto_len))
        nombre_acc = (correct_nombre/(transferencia_len+agregar_contacto_len))
        monto_acc = (correct_monto/(transferencia_len+agregar_contacto_len))
        entidad_acc = (correct_ent/(transferencia_len+agregar_contacto_len))
        cuenta_acc = (correct_cuenta/(transferencia_len+agregar_contacto_len))
        moneda_acc = (correct_moneda/(transferencia_len+agregar_contacto_len))
        doc_acc = (correct_doc/(transferencia_len+agregar_contacto_len))

    result = {
        "action_acc": action_acc,
        "alias_acc": alias_acc,
        "nombre_acc": nombre_acc,
        "monto_acc": monto_acc,
        "moneda_acc": moneda_acc,
        "cuenta_acc": cuenta_acc,
        "entidad_acc": entidad_acc,
        "doc_acc": doc_acc
    }

    return result

def compute_metrics_with_csv_building(save_to_csv=False, csv_path=None):
    def compute_metrics(eval_pred):
        predictions, labels, inputs = eval_pred
        decoded_preds = model_tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, model_tokenizer.pad_token_id)
        decoded_labels = model_tokenizer.batch_decode(labels, skip_special_tokens=True)

        inputs = np.where(inputs != -100, inputs, model_tokenizer.pad_token_id)
        decoded_inputs = model_tokenizer.batch_decode(inputs, skip_special_tokens=True)

        result = {}

        # Compute CER
        result["cer"] = cer.compute(predictions=decoded_preds, references=decoded_labels)
        
        # Compute Exact Match
        exact_match_res = exact_match.compute(predictions=decoded_preds, references=decoded_labels, ignore_case=True)
        result["exact_match"] = exact_match_res["exact_match"]

        # Compute Custom Eval
        result.update(custom_eval(predictions=decoded_preds, references=decoded_labels))

        if(result["exact_match"] < 1 and save_to_csv and csv_path is not None):
            non_matches = []
            for input, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                is_exact_match = pred.lower() == label.lower()
                if not is_exact_match:
                    non_matches.append({"frase": input, "reference_string": label, "predicted_string": pred})
            non_matches_df = pd.DataFrame(non_matches)
            non_matches_df.to_csv(csv_path, index=False)

        return {k: round(v, 4) for k, v in result.items()}
    return compute_metrics

cer = evaluate.load("cer", module_type="metric")
exact_match = evaluate.load("exact_match", module_type="metric")

modelo_base_path = "../models/base-spa-mt5"

model_tokenizer = MT5Tokenizer.from_pretrained(modelo_base_path, legacy=False)

base_model = MT5ForConditionalGeneration.from_pretrained(modelo_base_path)

data_collator = DataCollatorForSeq2Seq(tokenizer=model_tokenizer, model=base_model)