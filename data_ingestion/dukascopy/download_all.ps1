param(
  [string]$START_FROM = ""
)

$CSV_PATH = "instruments_all_from_readme.csv"
$TIMEFRAME = "d1"
$PRICE_TYPES = @("bid","ask")

$END_YEAR = (Get-Date).Year

$RETRIES = 3
$RETRY_PAUSE = 2000

$OUT_DIR = "download/raw"
$LOG_DIR = "logs"

New-Item -ItemType Directory -Force -Path $OUT_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null

$rows = Import-Csv $CSV_PATH | Sort-Object id

foreach ($row in $rows) {

    if ($START_FROM -ne "" -and $row.id -lt $START_FROM) {
        continue
    }

    $symbol = $row.id
    $startYear = ([datetime]$row.earliest_date_iso).Year

    Write-Host "==============================="
    Write-Host "Instrument: $symbol"
    Write-Host "From year : $startYear"
    Write-Host "==============================="

    for ($year = $startYear; $year -le $END_YEAR; $year++) {

        $from = "$year-01-01"
        $to   = "$year-12-31"

        foreach ($price in $PRICE_TYPES) {

            $expectedFile = Join-Path $OUT_DIR "$symbol-$TIMEFRAME-$price-$from-$to.csv"
            if (Test-Path $expectedFile) {
                Write-Host "SKIP (exists) $symbol | $TIMEFRAME | $price | $year"
                continue
            }

            $logFile = Join-Path $LOG_DIR "$symbol-$TIMEFRAME-$price-$year.log"

            Write-Host "→ $symbol | $TIMEFRAME | $price | $year"

            npx dukascopy-node `
              -i $symbol `
              -from $from `
              -to $to `
              -t $TIMEFRAME `
              -p $price `
              -f csv `
              -dir $OUT_DIR `
              -r $RETRIES `
              -rp $RETRY_PAUSE `
              -re `
              --silent `
              *> $logFile
        }
    }
}
