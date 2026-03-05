/// Tokenize a REPL input line into arguments using shell-like rules:
/// - Whitespace-separated tokens
/// - Double-quoted strings preserved as single arguments (quotes stripped)
/// - Single-quoted strings preserved as single arguments (quotes stripped)
/// - Backslash escaping within double-quoted strings
/// - Unicode within quotes passed through unmodified
pub fn tokenize(input: &str) -> Result<Vec<String>, String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut chars = input.chars().peekable();
    let mut in_double_quote = false;
    let mut in_single_quote = false;

    while let Some(ch) = chars.next() {
        if in_double_quote {
            if ch == '\\' {
                if let Some(&next @ ('"' | '\\')) = chars.peek() {
                    chars.next();
                    current.push(next);
                } else {
                    current.push(ch);
                }
            } else if ch == '"' {
                in_double_quote = false;
            } else {
                current.push(ch);
            }
        } else if in_single_quote {
            if ch == '\'' {
                in_single_quote = false;
            } else {
                current.push(ch);
            }
        } else {
            match ch {
                '"' => {
                    in_double_quote = true;
                }
                '\'' => {
                    in_single_quote = true;
                }
                ' ' | '\t' => {
                    if !current.is_empty() {
                        tokens.push(std::mem::take(&mut current));
                    }
                }
                _ => {
                    current.push(ch);
                }
            }
        }
    }

    if in_double_quote {
        return Err("Unterminated double quote".to_string());
    }
    if in_single_quote {
        return Err("Unterminated single quote".to_string());
    }

    if !current.is_empty() {
        tokens.push(current);
    }

    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_space_separated() {
        assert_eq!(
            tokenize("remember hello world").unwrap(),
            vec!["remember", "hello", "world"]
        );
    }

    #[test]
    fn double_quoted_string() {
        assert_eq!(
            tokenize(r#"remember "hello world" --importance 0.9"#).unwrap(),
            vec!["remember", "hello world", "--importance", "0.9"]
        );
    }

    #[test]
    fn single_quoted_string() {
        assert_eq!(
            tokenize("remember 'hello world' --importance 0.9").unwrap(),
            vec!["remember", "hello world", "--importance", "0.9"]
        );
    }

    #[test]
    fn escaped_quote_in_double_quotes() {
        assert_eq!(
            tokenize(r#"remember "he said \"hello\"" --importance 0.5"#).unwrap(),
            vec!["remember", r#"he said "hello""#, "--importance", "0.5"]
        );
    }

    #[test]
    fn empty_string_argument() {
        assert_eq!(
            tokenize(r#"remember "" --importance 0.5"#).unwrap(),
            vec!["remember", "--importance", "0.5"]
        );
    }

    #[test]
    fn empty_input() {
        assert_eq!(tokenize("").unwrap(), Vec::<String>::new());
    }

    #[test]
    fn only_whitespace() {
        assert_eq!(tokenize("   \t  ").unwrap(), Vec::<String>::new());
    }

    #[test]
    fn multiple_spaces_between_tokens() {
        assert_eq!(
            tokenize("remember   hello    world").unwrap(),
            vec!["remember", "hello", "world"]
        );
    }

    #[test]
    fn leading_trailing_whitespace() {
        assert_eq!(
            tokenize("  remember hello  ").unwrap(),
            vec!["remember", "hello"]
        );
    }

    #[test]
    fn unterminated_double_quote_error() {
        assert!(tokenize(r#"remember "hello"#).is_err());
    }

    #[test]
    fn unterminated_single_quote_error() {
        assert!(tokenize("remember 'hello").is_err());
    }

    #[test]
    fn unicode_content() {
        assert_eq!(
            tokenize("remember \"こんにちは世界\" --importance 0.8").unwrap(),
            vec!["remember", "こんにちは世界", "--importance", "0.8"]
        );
    }

    #[test]
    fn json_context_argument() {
        assert_eq!(
            tokenize(r#"remember "text" --context '{"key": "value"}'"#).unwrap(),
            vec!["remember", "text", "--context", r#"{"key": "value"}"#]
        );
    }

    #[test]
    fn tab_separated() {
        assert_eq!(
            tokenize("remember\thello\tworld").unwrap(),
            vec!["remember", "hello", "world"]
        );
    }

    #[test]
    fn backslash_outside_quotes() {
        assert_eq!(
            tokenize(r#"remember hello\nworld"#).unwrap(),
            vec![r#"remember"#, r#"hello\nworld"#]
        );
    }

    #[test]
    fn adjacent_quoted_and_unquoted() {
        // "hello"world becomes helloworld (shell semantics)
        assert_eq!(tokenize(r#""hello"world"#).unwrap(), vec!["helloworld"]);
    }
}
