import sys
from itertools import combinations

transaction_list = []
min_support_count = 0


def initialize_item_set():
    global transaction_list
    item_set_1 = {}
    for transaction in transaction_list:
        for item in transaction:
            if item in item_set_1:
                item_set_1[item] += 1
            else:
                item_set_1[item] = 1

    for item in list(item_set_1.keys()):
        if item_set_1[item] < min_support_count:
            del item_set_1[item]
    return item_set_1


def make_candidate_set_list(prev_keys_list, item_length):
    candidate_set_list = []
    if item_length == 2:
        tmp_candidates = list(combinations(prev_keys_list, item_length))
        for candidate in tmp_candidates:
            candidate_set_list.append(set(candidate))
    else:
        single_key_list = []
        for prev_keys in prev_keys_list:
            for prev_key in prev_keys:
                if prev_key not in single_key_list:
                    single_key_list.append(prev_key)
        tmp_candidates = list(combinations(single_key_list, item_length))
        for candidate in tmp_candidates:
            candidate_set_list.append(set(candidate))

    return candidate_set_list


def make_frequent_set(prev_keys_list, item_length, candidate_set_list):
    global transaction_list
    global min_support_count
    frequent_set_tmp = {}

    if item_length == 2:
        tmp_list = []
        # 밑에서 combination결과를 list로 감싸서 넣기 때문에 item 길이가 2일 경우 []로 감싸고 set으로 처리
        for prev_keys in prev_keys_list:
            tmp_list.append(set([prev_keys]))
        prev_keys_list = tmp_list
    else:
        tmp_list = []
        for prev_keys in prev_keys_list:
            tmp_list.append(set(prev_keys))
        prev_keys_list = tmp_list

    for candidate_set in candidate_set_list:
        count = 0
        for key in list(combinations(candidate_set, item_length - 1)):
            key = set(key)

            if key not in prev_keys_list:
                break
            count = count + 1
        # nCn-1 = n 임을 이용해 조건 체크
        if count == item_length:
            # unhashable type error로 key값을 tuple로 변경
            frequent_set_tmp[tuple(candidate_set)] = 0

    for key in frequent_set_tmp.keys():
        for transaction in transaction_list:
            if set(key).issubset(set(transaction)):
                frequent_set_tmp[key] += 1

    frequent_set = {key: frequent_set_tmp[key] for key in frequent_set_tmp.keys() if
                    frequent_set_tmp[key] >= min_support_count}
    return frequent_set


def apriori():
    global transaction_list
    item_set_total_list = []
    item_set_1 = initialize_item_set()
    item_set_total_list.append(item_set_1)

    item_length = 1
    while True:
        prev_keys_list = list(item_set_total_list[item_length - 1].keys())
        item_length += 1
        candidate_set_list = make_candidate_set_list(prev_keys_list, item_length)
        if not candidate_set_list:
            break
        frequent_set = make_frequent_set(prev_keys_list, item_length, candidate_set_list)
        if not frequent_set:
            break
        else:
            item_set_total_list.append(frequent_set)

    return item_set_total_list


def from_set_to_form(target_set):
    target_list = list(target_set)
    context = ','.join(str(target) for target in target_list)
    text = "{" + context + "}"
    return text


def make_output_text_by_association(item_set_total_list):
    global transaction_list
    num_of_transaction = len(transaction_list)
    output_file_text = []

    for list_index in range(1, len(item_set_total_list)):
        for item_set, item_set_count in item_set_total_list[list_index].items():
            sub_itemset_length = list_index
            while sub_itemset_length > 0:
                sub_itemset = list(combinations(item_set, sub_itemset_length))
                for item in sub_itemset:
                    associative_item = set(item_set).difference(item)
                    support = (int(item_set_count) / num_of_transaction) * 100

                    item_count = 0
                    for transaction in transaction_list:
                        if set(item).issubset(transaction):
                            item_count += 1
                    confidence = (int(item_set_count) / item_count) * 100

                    output_file_text.append(
                        '{}\t{}\t{:.2f}\t{:.2f}\n'.format(from_set_to_form(item), from_set_to_form(associative_item),
                                                          support, confidence))

                sub_itemset_length -= 1

    return output_file_text


def main():
    global transaction_list
    global min_support_count
    min_support = int(sys.argv[1])
    input_file_name = sys.argv[2]
    output_file_name = sys.argv[3]

    input_file = open("./" + input_file_name, 'r')
    transaction_list_raw = input_file.readlines()
    input_file.close()

    transaction_count = 0
    for transaction in transaction_list_raw:
        refined_transaction = transaction.replace("\n", "").split('\t')
        transaction_list.append(refined_transaction)
        transaction_count += 1

    min_support_count = int(transaction_count * min_support / 100)
    item_set_total_list = apriori()
    output_file_context = make_output_text_by_association(item_set_total_list)

    output_file = open("./" + output_file_name, 'w')
    for context in output_file_context:
        output_file.write(context)
    output_file.close()


if __name__ == "__main__":
    main()
