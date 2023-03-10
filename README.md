This is a test repo to work out issues with pygmalion that I am having.
I am seeing the model generate text that is often:
- only a newline (\n)
- prepended with a space or newline before a result
- Gets <BOT>: as the characters name

Some changes that improve the results:
- Remove newlines between <STARTS>
- Remove space after `Haruka:` prompt
- Remove newline after persona
