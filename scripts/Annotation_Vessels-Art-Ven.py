from mevis import *

def SavedMask ():
  #print(ctx.field("SavingInfo").stringValue())
  #print(ctx.field("itkImageFileWriter.info").stringValue())
  ctx.field("SavingInfo").setStringValue(ctx.field("itkImageFileWriter.info").stringValue())
  return

def LoadData ():
  ctx.field("SavingInfo").setStringValue("")
  MLAB.processEvents()
  return

