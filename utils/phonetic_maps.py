# utils/phonetic_maps.py

IPA_TO_XSAMPA = {
    # Vogais orais e nasais básicas
    "a": "a", "ã": "a~", "e": "e", "ẽ": "e~", "i": "i", "ĩ": "i~", 
    "o": "o", "õ": "o~", "u": "u", "ũ": "u~",
    
    # Vogais abertas, médias e centrais
    "ɐ": "6", "ɐ̃": "6~", "ɔ": "O", "ɔ̃": "O~", "ə": "@", "ə̃": "@~", 
    "ɛ": "E", "ɛ̃": "E~", "ɨ": "1", "ɨ̃": "1~", "ɪ": "I", "ʊ": "U", "ʌ̃": "V~",
    
    # Consoantes padrão
    "b": "b", "c": "c", "d": "d", "f": "f", "g": "g", "h": "h", 
    "j": "j", "k": "k", "l": "l", "m": "m", "n": "n", "p": "p", 
    "r": "r", "s": "s", "t": "t", "v": "v", "w": "w", "x": "x", "z": "z",
    
    # Consoantes especiais e fricativas/oclusivas complexas
    "ŋ": "N", "ð": "D", "ɣ": "G", "ɦ": "h\\", "ɫ": "5", "ɲ": "J", 
    "ɹ": "r\\", "ɽ": "r\\`", "ɾ": "4", "ʀ": "R", "ʁ": "R\\", 
    "ʃ": "S", "ʎ": "L", "ʒ": "Z", "θ": "T", "χ": "X",
    
    # Sons longos, labializados e aspirados compostos
    "aː": "a:", "bʷ": "b_w", "dː": "d:", "eː": "e:", "fʷ": "f_w", 
    "gʷ": "g_w", "iː": "i:", "kʰ": "k_h", "kʷ": "k_w", "oː": "o:", 
    "pʷ": "p_w", "sː": "s:", "tʰ": "t_h", "vʷ": "v_w", "ɛː": "E:"
}

def convert_to_xsampa(formatted_output):
    tokens = formatted_output.split('|')
    converted_tokens = []
    for token in tokens:
        if token in IPA_TO_XSAMPA:
            converted_tokens.append(IPA_TO_XSAMPA[token])
        else:
            converted_tokens.append(token)
    return '|'.join(converted_tokens)