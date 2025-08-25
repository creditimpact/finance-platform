"""Sample Account Detail text blocks for parser tests."""

BLOCK_WITH_DIGITS = """PALISADES FU
Account #            123             456             789
Payment Status:      OK              OK              OK
"""

BLOCK_WITHOUT_DIGITS = """PALISADES FU
Account #            t disputed      ****            N/A
Balance: 0
"""

BLOCK_WITH_MASKED = """PALISADES FU
Account #            ****1234        12 34 56        1234-5678-9012
Balance: 0
"""

BLOCK_WITH_COLLECTION_STATUS = """PALISADES FU
TransUnion          Experian          Equifax
Account #            123             456             789
Payment Status:      Collection/Chargeoff  Collection/Chargeoff  Charge-Off
"""

