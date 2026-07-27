param(
    [string]$Output = "Supplementary_Data_Generated_Peptides.csv",
    [int]$CandidateLimit = 1000
)

$ErrorActionPreference = "Stop"

Add-Type -TypeDefinition @"
using System;
using System.Collections.Generic;
using System.Linq;

public static class NearestPeptideIdentity
{
    private const string Valid = "ACDEFGHIKLMNPQRSTVWY";

    public static string Clean(string seq)
    {
        if (seq == null) return "";
        var chars = new List<char>();
        foreach (var c0 in seq.Trim().ToUpperInvariant())
        {
            if (Valid.IndexOf(c0) >= 0) chars.Add(c0);
        }
        return new string(chars.ToArray());
    }

    public static int EditDistance(string a, string b)
    {
        if (a == b) return 0;
        if (a.Length < b.Length)
        {
            var tmp = a; a = b; b = tmp;
        }
        int[] prev = new int[b.Length + 1];
        for (int j = 0; j <= b.Length; j++) prev[j] = j;
        for (int i = 1; i <= a.Length; i++)
        {
            int[] cur = new int[b.Length + 1];
            cur[0] = i;
            for (int j = 1; j <= b.Length; j++)
            {
                int sub = prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1);
                int del = prev[j] + 1;
                int ins = cur[j - 1] + 1;
                cur[j] = Math.Min(Math.Min(del, ins), sub);
            }
            prev = cur;
        }
        return prev[b.Length];
    }

    public static double IdentityPct(string query, string target)
    {
        int denom = Math.Max(Math.Max(query.Length, target.Length), 1);
        return 100.0 * (1.0 - ((double)EditDistance(query, target) / denom));
    }

    public static string[] FindNearest(string[] queries, string[] training, int limit)
    {
        var kmerIndex = new Dictionary<string, List<int>>();
        var lengthIndex = new Dictionary<int, List<int>>();
        for (int idx = 0; idx < training.Length; idx++)
        {
            string seq = training[idx];
            if (!lengthIndex.ContainsKey(seq.Length)) lengthIndex[seq.Length] = new List<int>();
            lengthIndex[seq.Length].Add(idx);
            var seen = new HashSet<string>();
            for (int i = 0; i <= seq.Length - 3; i++) seen.Add(seq.Substring(i, 3));
            foreach (var kmer in seen)
            {
                if (!kmerIndex.ContainsKey(kmer)) kmerIndex[kmer] = new List<int>();
                kmerIndex[kmer].Add(idx);
            }
        }

        string[] results = new string[queries.Length];
        for (int qi = 0; qi < queries.Length; qi++)
        {
            string q = queries[qi];
            var candidates = new HashSet<int>();
            var counts = new Dictionary<int, int>();

            for (int len = Math.Max(1, q.Length - 2); len <= q.Length + 2; len++)
            {
                if (lengthIndex.ContainsKey(len))
                {
                    foreach (var idx in lengthIndex[len]) candidates.Add(idx);
                }
            }

            for (int i = 0; i <= q.Length - 3; i++)
            {
                string kmer = q.Substring(i, 3);
                if (!kmerIndex.ContainsKey(kmer)) continue;
                foreach (var idx in kmerIndex[kmer])
                {
                    candidates.Add(idx);
                    counts[idx] = counts.ContainsKey(idx) ? counts[idx] + 1 : 1;
                }
            }

            if (candidates.Count == 0)
            {
                for (int i = 0; i < Math.Min(limit, training.Length); i++) candidates.Add(i);
            }

            var ranked = candidates
                .OrderByDescending(idx => counts.ContainsKey(idx) ? counts[idx] : 0)
                .ThenBy(idx => Math.Abs(training[idx].Length - q.Length))
                .Take(limit);

            string bestSeq = "";
            double bestIdentity = -1.0;
            foreach (var idx in ranked)
            {
                double identity = IdentityPct(q, training[idx]);
                if (identity > bestIdentity)
                {
                    bestIdentity = identity;
                    bestSeq = training[idx];
                    if (bestIdentity >= 100.0) break;
                }
            }
            results[qi] = bestIdentity.ToString("0.###") + "`t" + bestSeq;
        }
        return results;
    }
}
"@

function Get-SequenceColumn($Rows) {
    $names = $Rows[0].PSObject.Properties.Name
    foreach ($candidate in @("sequence", "Sequence", "seq", "Seq", "AASequence")) {
        if ($names -contains $candidate) { return $candidate }
    }
    throw "No sequence column found."
}

function Read-GeneratedRows($Path, $Mode) {
    if (!(Test-Path $Path)) { return @() }
    $rows = @(Import-Csv $Path)
    if ($rows.Count -eq 0) { return @() }
    $seqCol = Get-SequenceColumn $rows
    foreach ($row in $rows) {
        $seq = [NearestPeptideIdentity]::Clean([string]$row.$seqCol)
        if ($seq.Length -eq 0) { continue }
        $row | Add-Member -NotePropertyName "clean_sequence" -NotePropertyValue $seq -Force
        $row | Add-Member -NotePropertyName "generation_mode_added" -NotePropertyValue $Mode -Force
        $row
    }
}

function Read-TrainingSequences {
    $paths = @(
        "Dataset/1ststage/unlabelled_positive.csv",
        "Dataset/1ststage/unlabelled_negative.csv",
        "Dataset/1ststage/mic_data.csv",
        "Dataset/1ststage/Uniprot_0_25_train.csv",
        "Dataset/2ndstage/Train_data.csv",
        "Dataset/3rdstage/multilabel_dataset.csv"
    )
    $set = [System.Collections.Generic.HashSet[string]]::new()
    foreach ($path in $paths) {
        if (!(Test-Path $path)) { continue }
        $rows = @(Import-Csv $path)
        if ($rows.Count -eq 0) { continue }
        $seqCol = Get-SequenceColumn $rows
        foreach ($row in $rows) {
            $seq = [NearestPeptideIdentity]::Clean([string]$row.$seqCol)
            if ($seq.Length -gt 0) { [void]$set.Add($seq) }
        }
    }
    return @($set)
}

function Number-OrBlank($Value) {
    if ($null -eq $Value -or "$Value" -eq "") { return "" }
    return $Value
}

function Clip-Reward($Amp, $Mic) {
    $a = 0.0
    $m = 0.0
    [void][double]::TryParse([string]$Amp, [ref]$a)
    [void][double]::TryParse([string]$Mic, [ref]$m)
    return [Math]::Min(1.0, [Math]::Max(-1.0, $a + $m))
}

$stage3Rows = @()
$stage3Rows += Read-GeneratedRows "Result_1ststage/generated_denovo_ecd_ar_tagged_amp_22func_pred.csv" "de_novo"
$stage3Rows += Read-GeneratedRows "Result_1ststage/generated_analogs_ecd_ar_tagged_amp_22func_pred.csv" "analog"
$stage3Rows += Read-GeneratedRows "Final_Result/generated_denovo_ecd_ar_tagged_amp_22func_pred.csv" "de_novo"
$stage3Rows += Read-GeneratedRows "Final_Result/generated_analogs_ecd_ar_tagged_amp_22func_pred.csv" "analog"

if ($stage3Rows.Count -eq 0) {
    throw "No final Stage-3 generated peptide prediction files were found."
}

$trainingSeqs = Read-TrainingSequences
if ($trainingSeqs.Count -eq 0) {
    throw "No training sequences found for nearest-neighbor identity calculation."
}

Write-Host "Loaded $($stage3Rows.Count) final generated peptides."
Write-Host "Loaded $($trainingSeqs.Count) unique training sequences."
Write-Host "Computing nearest-neighbor identity..."

$queries = [string[]]($stage3Rows | ForEach-Object { $_.clean_sequence })
$nearest = [NearestPeptideIdentity]::FindNearest($queries, [string[]]$trainingSeqs, $CandidateLimit)

$probCols = @($stage3Rows[0].PSObject.Properties.Name | Where-Object { $_ -like "*_prob" -and $_ -ne "amp_prob" })
$mlProbCols = @($probCols | Sort-Object)

$outRows = for ($i = 0; $i -lt $stage3Rows.Count; $i++) {
    $row = $stage3Rows[$i]
    $parts = $nearest[$i] -split "`t", 2
    $identity = $parts[0]
    $nearestSeq = if ($parts.Count -gt 1) { $parts[1] } else { "" }
    $mode = if ($row.PSObject.Properties.Name -contains "generation_mode") { $row.generation_mode } else { $row.generation_mode_added }
    $reward = if ($row.PSObject.Properties.Name -contains "reward_score") { $row.reward_score } else { Clip-Reward $row.amp_prob $row.mic_prob_low }

    $ordered = [ordered]@{
        peptide_id = $row.name
        sequence = $row.clean_sequence
        generation_mode = $mode
        seed = if ($row.PSObject.Properties.Name -contains "seed") { $row.seed } else { "" }
        reward_score = $reward
        amp_probability = $row.amp_prob
        mic_class_probability = $row.mic_prob_low
        ensemble_screening_score = $row.ensemble_softscore
        ensemble_amp = $row.ensemble_amp
        length = $row.length
        hydrophobicity = $row.hydrophobicity
        hydrophobic_moment = $row.hydrophobic_moment
        charge = $row.charge
        isoelectric_point = $row.isoelectric_point
    }

    foreach ($col in $mlProbCols) {
        $name = "ML_A2P_" + (($col -replace "_prob$", "") -replace "[^0-9A-Za-z_]", "_") + "_probability"
        $ordered[$name] = $row.$col
    }

    $ordered["hemolytic_probability"] = Number-OrBlank $row.Hemolytic_prob
    $ordered["hemolytic_prediction"] = Number-OrBlank $row.Hemolytic
    $ordered["cytotoxic_probability"] = Number-OrBlank $row.Cytotoxic_prob
    $ordered["cytotoxic_prediction"] = Number-OrBlank $row.Cytotoxic
    $ordered["nearest_training_sequence"] = $nearestSeq
    $ordered["nearest_neighbor_identity_to_training_pct"] = $identity

    [pscustomobject]$ordered
}

$outRows | Export-Csv $Output -NoTypeInformation
Write-Host "Wrote $Output with $($outRows.Count) rows."
