$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskSource = (Get-ChildItem -LiteralPath (Join-Path $taskRoot 'paper') -Filter '*v313*.docx' | Select-Object -First 1).FullName
$taskPdf = Join-Path $taskRoot 'tmp\v313_paper_qa\DCT_v313_qa.pdf'
# Native CLR arguments avoid PowerShell's boxed enum/VARIANT conversion.
Add-Type -TypeDefinition @'
using System;
using System.Reflection;
public static class V313WordRender {
    static object Call(object target, string name, BindingFlags flags, params object[] args) {
        return target.GetType().InvokeMember(name, flags, null, target, args);
    }
    public static int Run(string source, string output) {
        object word = null, doc = null;
        try {
            word = Activator.CreateInstance(Type.GetTypeFromProgID("Word.Application"));
            Call(word, "Visible", BindingFlags.SetProperty, false);
            Call(word, "DisplayAlerts", BindingFlags.SetProperty, 0);
            object docs = Call(word, "Documents", BindingFlags.GetProperty);
            doc = Call(docs, "Open", BindingFlags.InvokeMethod, source, false, true);
            Call(doc, "Repaginate", BindingFlags.InvokeMethod);
            try {
                Call(doc, "ExportAsFixedFormat", BindingFlags.InvokeMethod,
                     output, 17, false, 0, 0, 1, 1, 0, true, true, 0, true, true, false);
            } catch {
                Call(doc, "SaveAs2", BindingFlags.InvokeMethod, output, 17);
            }
            return Convert.ToInt32(Call(doc, "ComputeStatistics", BindingFlags.InvokeMethod, 2));
        } finally {
            if (doc != null) Call(doc, "Close", BindingFlags.InvokeMethod, 0);
            if (word != null) Call(word, "Quit", BindingFlags.InvokeMethod);
        }
    }
}
'@
$taskPages = [V313WordRender]::Run($taskSource, $taskPdf)
Write-Output ('Pages: ' + $taskPages)
Write-Output ('QA PDF: ' + $taskPdf)
