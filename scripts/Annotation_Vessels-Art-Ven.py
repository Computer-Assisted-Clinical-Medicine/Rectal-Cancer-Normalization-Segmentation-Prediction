from mevis import *

def ChangedModeArteries ():
  if ctx.field("modeArteries").value == 0:
    ctx.field("voxelWriteValueArteries").value = 0
  else:
    ctx.field("voxelWriteValueArteries").value = 1
    
  #print(ctx.field("modeArteries").value)
  #print(ctx.field("voxelWriteValueArteries").value)
  return

def SavedMask ():
  #print(ctx.field("SavingInfo").stringValue())
  #print(ctx.field("itkImageFileWriter.info").stringValue())
  ctx.field("SavingInfo").setStringValue(ctx.field("itkImageFileWriter.info").stringValue())
  return

def LoadedData ():
  ctx.field("SavingInfo").setStringValue("")
  return

